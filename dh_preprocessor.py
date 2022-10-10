import os
import pickle
from tqdm import tqdm, trange
from transformers import GPT2Tokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch

'''커스텀 데이터 처리 스크립트. writing prompt 형식으로 되어있는 데이터셋을 progen 형식으로 변경합니다. 
데이터는 각각 save_dir 내에 {split}.{data_type}_source 형태로 들어 있어야 합니다.'''

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

def wp():
    save_dir = 'data/wp'
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    split_size = {
        'train': 10000,
        'valid': 5000,
        'test': 1000
    }

    for split in ['test']:
        src_lines = open(f'{save_dir}/{split}.wp_source').readlines()
        tgt_lines = open(f'{save_dir}/{split}.wp_target').readlines()

        examples = []
        for src, tgt in tqdm(zip(src_lines, tgt_lines),
                             desc=split, total=len(tgt_lines)):
            src= src.strip().replace('<newline>', '\n')
            tgt = tgt.strip().replace('<newline>', '\n')

            if len(tokenizer.tokenize(
                    f'{src} [SEP] {tgt} <|endoftext|>')) > 1024 and split in ("train", "valid"):
                    continue

            examples.append({
                'condition': src,
                'text': tgt
            })

            if len(examples) >= split_size[split]:
                break

        print(f'#{split}: {len(examples)}')
        pickle.dump(examples, open(f'{save_dir}/{split}.pickle', 'wb'))

def cnn():
    # load model
    # simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
    # simcse_model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large").cuda()


    output_dir = 'data/cnn'
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    split_size = {
        'train': 10000,
        'valid': 5000,
        'test': 1000
    }

    for split in ['test']:
        src_lines = open(f'{output_dir}/{split}.cnndm_source').readlines()
        tgt_lines = open(f'{output_dir}/{split}.cnndm_target').readlines()

        examples = []
        for src, tgt in tqdm(zip(src_lines, tgt_lines),
                             desc=split, total=len(tgt_lines)):
            src = "\n\n".join(sent_tokenize(src.strip()))
            tgt = "\n\n".join(sent_tokenize(tgt.strip()))



            if len(tokenizer.tokenize(
                    f'{src} [SEP] {tgt} <|endoftext|>')) > 1024 and split in ("train", "valid"):
                continue


            examples.append({
                'condition': src,
                'text': tgt
            })

            if len(examples) >= split_size[split]:
                break

        print(f'#{split}: {len(examples)}')
        pickle.dump(examples, open(f'{output_dir}/{split}.pickle', 'wb'))

if __name__ == "__main__":
    cnn()
    wp()