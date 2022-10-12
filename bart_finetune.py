import os
import fire
import pickle
import wandb

from models.bart import BART

BATCH_SIZE = 16
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1


def get_vocab(dataset, vocab_size):
    if vocab_size == 'null':
        return None
    return pickle.load(open(f'data/{dataset}/vocab_{vocab_size}.pickle', 'rb'))


def load_data(dataset, split, vocab, keep_condition):
    if vocab in ['null', 'full']:
        examples = pickle.load(open(f'data/{dataset}/{split}.pickle', 'rb'))
        if vocab == 'null':
            return [example['condition'] for example in examples]
        else:
            return [example['text'] for example in examples]
    else:
        examples = pickle.load(open(
            f'data/{dataset}/extracted_{split}_{vocab}words.pickle', 'rb'))
        if keep_condition:
            return [example['condition'] + ' [SEP] ' + example['extracted_text']
                    for example in examples]
        else:
            return [example['extracted_text'] for example in examples]

def load_tgt_relevance(dataset, split, tgt_vocab):
    examples = pickle.load(open(f'data/{dataset}/{split}.pickle', 'rb'))
    if tgt_vocab == 'full':
        return [example['tgt_relevance'] for example in examples]
    return None


def main(dataset='cnn',
         src_vocab='0.25',
         tgt_vocab='full',
         n_epochs=10,
         wandb_pj_name="progen"):
    if wandb_pj_name:
        wandb.init(entity='lucas01',project=wandb_pj_name, name=f"{dataset}-{src_vocab}-{tgt_vocab}")

    if os.path.exists(f'training_logs/bart_{dataset}_{src_vocab}-{tgt_vocab}'):
        print('Training path existed! Remove it if wanna re-train.')
        return

    bart = BART()

    for split in ['train', 'valid']:
        src_texts = load_data(dataset, split, src_vocab, keep_condition=True)
        tgt_texts = load_data(dataset, split, tgt_vocab, keep_condition=False)
        tgt_relevances = load_tgt_relevance(dataset, split, tgt_vocab)
        bart.load_data(set_type=split, src_texts=src_texts, tgt_texts=tgt_texts, tgt_relevances=tgt_relevances)

    train_steps = n_epochs * (len(bart.dataset['train']) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    bart.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    bart.create_training_log(
        eval_steps=len(bart.dataset['train']) // BATCH_SIZE // 2, # 1 epoch에 evaluation set에 대해 2번 평가하기.
        label=f'bart_{dataset}_{src_vocab}-{tgt_vocab}')

    noise_vocab = get_vocab(dataset, src_vocab)
    for epoch in range(n_epochs):
        bart.train_epoch(batch_size=BATCH_SIZE, noise_vocab=noise_vocab, tgt_vocab=tgt_vocab)


if __name__ == '__main__':
    fire.Fire(main)
