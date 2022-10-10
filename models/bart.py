from collections import namedtuple
import random
from tqdm import tqdm, trange
import os
import nltk
import wandb
import torch
from nltk.tokenize import sent_tokenize

from fairseq.sequence_generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer

from .bart_utils import BARTModelWrapper
from .bart_utils import try_wandb_log, try_wandb_add_example



BART_MAX_LEN = 1024


TextPairData = namedtuple('TextPairData', ['src_text', 'tgt_text'])


@torch.no_grad()
def compute_sim_score(simcse_model, simcse_tokenizer, prompt, txt):
    r_id = simcse_tokenizer.encode(prompt)
    h_id = simcse_tokenizer.encode(txt)

    r_id, h_id = r_id[:min(len(r_id), 512)], h_id[:min(len(h_id), 512)] # truncate to 512
    ref_emb = simcse_model(input_ids = torch.LongTensor(r_id).view(1,-1).cuda())
    hypo_emb = simcse_model(input_ids = torch.LongTensor(h_id).view(1,-1).cuda())
    dotprod = (ref_emb.pooler_output * hypo_emb.pooler_output).sum(-1)
    ref_norm = torch.norm(ref_emb.pooler_output, p=2)
    hypo_norm = torch.norm(hypo_emb.pooler_output, p=2)
    sim = dotprod / (ref_norm * hypo_norm) 
    return sim.detach().cpu().item()

class BART:
    def __init__(self):
        self._model = BARTModelWrapper()
        # self._simcse = AutoModel.from_pretrained("")
        self._simcse_model = AutoModel.from_pretrained("/princeton-nlp/unsup-simcse-roberta-large").cuda()
        self._simcse_tokenizer = AutoTokenizer.from_pretrained("/princeton-nlp/unsup-simcse-roberta-large")
        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0
        self._ex_table = wandb.Table(columns=["step","src","gen","tgt"])

        self._dataset = {}
        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'training_logs/{label}'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'ckpt_gens'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

    def load_data(self, set_type, src_texts, tgt_texts, tgt_relevances):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text, tgt_relevance in zip(src_texts, tgt_texts, tgt_relevances):
            self._dataset[set_type].append(TextPairData(
                src_text=src_text, tgt_text=tgt_text))

    def train_epoch(self, batch_size, noise_vocab, tgt_vocab):
        assert 'train' in self._dataset
        self.get_seq2seq_loss = self._get_seq2seq_loss_w_rel if tgt_vocab == "full" else self._get_seq2seq_loss

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._model.split_to_gpus(2)
            self._model.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for example in batch:
                noised_src_text = self._add_noise(example.src_text, noise_vocab)

                loss = self.get_seq2seq_loss(
                    src_text=noised_src_text, tgt_text=example.tgt_text)
                loss = loss / batch_size
                try_wandb_log({"train/loss":loss.detach().item()}, step=self._global_step)
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1
            if self._global_step % self._eval_steps == 0:
                self.gen_log()

    def evaluate(self):
        assert 'valid' in self._dataset
        self._model.split_to_gpus(1)
        self._model.eval()

        loss_list = []
        for example in tqdm(self._dataset['valid'], desc='Evaluating'):
            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_text=example.src_text, tgt_text=example.tgt_text)

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, cond, top_k, top_p):
        self._model.split_to_gpus(1)
        self._model.eval()

        generator = SequenceGenerator(
            tgt_dict=self._model.dictionary,
            max_len_b=BART_MAX_LEN,
            sampling=True,
            sampling_topk=top_k,
            sampling_topp=top_p)

        src_tokens = self._model.encode(cond)[:BART_MAX_LEN]

        outputs = generator.generate(
            models=[self._model.model],
            sample={'net_input': {
                'src_tokens': src_tokens.unsqueeze(0).to('cuda'),
                'src_lengths': torch.tensor([len(src_tokens)]).to('cuda')
            }})

        return self._model.decode(outputs[0][0]['tokens'].cpu())

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)
        

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        try_wandb_log({"dev/loss":eval_loss, 'dev/best_loss':self._best_dev_loss}, step=self._global_step)
        self._log_file.flush()

        generation_file = open(
            f'{self._log_dir}/ckpt_gens/step{self._global_step}.txt', 'w')

        for example in self._dataset['valid'][:10]:
            gen_text = self.generate(example.src_text, top_k=-1., top_p=0.95)

            print('SOURCE:\n', example.src_text, '\n', '-' * 50, '\n',
                  'GENERATION:\n', gen_text, '\n', '-' * 50, '\n',
                  'TARGET:\n', example.tgt_text, '\n', '=' * 100, '\n\n\n',
                  file=generation_file)
            try_wandb_add_example(self._ex_table, self._global_step,example.src_text, gen_text, example.tgt_text)
            generation_file.flush()

    def _get_seq2seq_loss(self, src_text, tgt_text):
        src_tokens = self._model.encode(src_text)[:BART_MAX_LEN].unsqueeze(0)
        tgt_tokens = self._model.encode(tgt_text)[:BART_MAX_LEN].unsqueeze(0)

        logits, extra = self._model(
            src_tokens=src_tokens,
            src_lengths=torch.tensor([src_tokens.shape[1]]),
            prev_output_tokens=tgt_tokens)

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._model.dictionary.pad())
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
    
    def _get_seq2seq_loss_w_rel(self, src_text , tgt_text, method:str = "direct"):
        '''
        tgt_relevance(Dict[str, float]) : 어디에서 어디까지(key, i.e. 0-14)가 relevance가 얼마인지(value) 정보를 가지고 있음.
        method(str, either "direct" or "mmd") : 각 시퀀스의 relevance 정보를 이용해서 직접 최적화할지, 아니면 MMD로 loosly 최적화할지 선택. 
        '''
        # compute relevance of simcse on-the-fly
        try:
            prompt = src_text.split(' [SEP] ')[0]
        except:
            pass
        tgt_relevance = {}
        offset = 0
        for sent in sent_tokenize(src_text): # sent_tokenize strips all \n\n
            _len = self._model.encode(sent).size(-1)-2 # exclude </s> and <s>
            if offset + _len > BART_MAX_LEN: break
            tgt_relevance[f"{offset}-{offset+_len}"] = compute_sim_score(self._simcse_model,self._simcse_tokenizer, prompt, sent)
            offset += (_len + 2) # +2 because of \n\n
        

        src_tokens = self._model.encode(src_text)[:BART_MAX_LEN].unsqueeze(0)
        tgt_tokens = self._model.encode(tgt_text)[:BART_MAX_LEN].unsqueeze(0)

        logits, extra = self._model(
            src_tokens=src_tokens,
            src_lengths=torch.tensor([src_tokens.shape[1]]),
            prev_output_tokens=tgt_tokens)

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._model.dictionary.pad())
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        ### relevance distribution matching loss ###
        assert method == "direct" 
        z_p = extra["last_encoder_hidden"].mean(dim=1)
        z_s = extra["last_decoder_hidden"]
        rel_losses = []
        # for idx, item in enumerate(tgt_relevance):
        for pos, val in tgt_relevance.items():
            start, end = map(int, pos.split("-"))
            z_si = z_s[0, start:end].mean(dim=1)
            val = float(val)
            _mse_loss = (val - z_p.dot(z_si)/torch.norm(z_p)/torch.norm(z_si)).pow(2)
            rel_losses.append(_mse_loss)
        rel_losses = torch.cat(rel_losses).mean()

        loss += 0.1*rel_losses

        return loss

    def _add_noise(self, text, noise_vocab):
        if text.find(' [SEP] ') == -1:
            return text
        prompt = text.split(' [SEP] ')[0]
        template = text.split(' [SEP] ')[1]

        noised_paras = []
        for para in template.split('\n'):
            words = nltk.word_tokenize(para)
            for i in range(len(words)):
                if random.random() < 0.1:
                    words[i] = random.choice(noise_vocab)
                    if i + 1 < len(words) and random.random() < 0.5:
                        words[i + 1] = random.choice(noise_vocab)
                        if i + 2 < len(words) and random.random() < 0.5:
                            words[i + 2] = random.choice(noise_vocab)
                            if i + 3 < len(words) and random.random() < 0.5:
                                words[i + 3] = random.choice(noise_vocab)

            noised_paras.append(' '.join(words))

        result = prompt + ' [SEP] ' + '\n'.join(noised_paras)

        return result

    @property
    def dataset(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()
