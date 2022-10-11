import os
import pickle
from tqdm import tqdm, trange
from transformers import GPT2Tokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import multiprocessing as mp

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

def compute_tgt_relevance(simcse_model, simcse_tokenizer, bart_tokenizer, prompt, tgt):
    '''학습에 사용되는 bart tokenizer를 이용해서 tgt의 sentence index를 찾고 relevance score를 계산하여 dictionary로 반환합니다.'''
    bart_tok_enc, bart_tok_dec = bart_tokenizer
    
    s_indices = []
    sim_scores = []
    start = 0
    for tgt_s in sent_tokenize(tgt):
        s_tok = bart_tok_enc(tgt_s)
        end = start + len(s_tok) - 2
        s_indices.append((start, end))
        # \n\n -> +2
        start = end + 2

        # compute_simcse scores
        sim_score = compute_sim_score(simcse_model, simcse_tokenizer, prompt, tgt_s)
        sim_scores.append(sim_score)


    # for debugging

    # tgt_tok = bart_tok_enc(tgt)
    # print(tgt_tok)
    # for s,e in s_indices:
    #     print(tgt_tok[s:e])
    #     decoded = bart_tok_dec(tgt_tok[s:e])
    #     print(decoded)

    return {f"{i}-{j}":k for (i,j), k in zip(s_indices, sim_scores)}


''' 문장단위로 한칸씩 앞으로 잘리는 것 확인.
tensor([    0, 27696,   377,    71,    39,   744,     6,  3188,  6718,  9239,
        10124,   991,    34,  1747,   303,    10, 18403,   317,     4, 50118,
        50118,   133,   320,  1482,     9,    22,  4310,  3107,   843,   113,
           21,  8203,   719,   545,    23, 24534,   241, 38314, 23433,     6,
           10, 16233,    11, 26067,     6,  8683,     6,  1437,  1044,     9,
        16233,   942,  9795,  5387,  1469,   289,  2678,  8476,   174,  3480,
           15,   294,     4, 50118, 50118,  1708,     5, 17353,    34,    45,
        36771,    10,   251,    12, 12364,   284, 13824,     4, 50118, 50118,
          113,   713,   662,   127,   284,     8,    38,  2435,    14,   127,
        13404,    18, 14202,  1141,  5363,  4171, 10124,   991,     8,    49,
         1354,  7616,  2764,  7456,    10, 16233,    11,  8683,    88, 34711,
          127, 13404,    89,    60,    39,  1354,  9508,  1069, 10124,   991,
           26,    15,    69,   622,  1842,     4, 50118, 50118,   113,  8170,
           19, 12389,     5,  5430,  1666,    31,   127,  1150,    18,   964,
            8,   284,  9052,    14,    37,   770,     7,    28,  8203,    11,
            5, 27925, 37804,     6,     5,   247,    11,    61,    37,    21,
         2421,     8,  1179,     6,    39,  8605,    58,     6,   683,   456,
            6,  8266,    30,    39,  9515, 22724,  2650,  1141,    72, 50118,
        50118, 35689, 10124,   991,   115,    45,    28,  1348,    13,  1129,
            4, 50118, 50118,   530,   281,   991,    18,  4194,   408,    31,
           39,    78,  3397,     8,    39,  1141,     9,    55,    87,   389,
          107, 12248,    11,   461,    11,     5,   377,   137, 10124,   991,
          962,   502,   379,    23,  1046,  7383,     4, 50118, 50118,   530,
          254,  1069, 10124,   991,    21,   576,    10, 23923,  3629,  4128,
           81,    69,  1150,    18,   474,     6,    61,  5363, 10124,   991,
         8960,  4951,    11,   461,     6,  4564,    39,   408,    58, 30088,
         3558,    49,  1150,    18,   301,     4, 50118, 50118, 17206, 10124,
          991,    18,   744,     6,  5363, 10124,   991,  2928,    69,    10,
         7022,  1623,    31,    10,   575,  2122,    11,  2005, 12811,     6,
          886,     6,     8,   362,   123,     7,  1095,    23,     5,   184,
            9,   284,   964,    11,   663,   194,     4, 50118, 50118,  4993,
        10124,   991,   962,     6,   284,   453,  7311,    51,   399,    75,
          216,   147,     5,   809,    21,  2034,     4, 50118, 50118, 36525,
         6113,  8650,     6,     5,   285,   661,    13,  9508,  1069, 10124,
          991,     6,    26,    94,  1035,    14,  5363, 10124,   991,  2928,
           69,  1623,    18,   809,    31,    10,  6172,   184,    11,   663,
            4, 50118, 50118,  4148,   302,     6,    37,    26,     6,    22,
        30770,     6,     5,   284,    16,    45,  1372,     8,    51,  1346,
           14,  5363,    64, 24569,   123, 11263,    79,  1072,     6,   190,
          114,    24,    16,    45,    99,  9239,   770,     4, 50118, 50118,
         1708,     5,  1159,  1437,    32,    23,  1987,  4730,    14,    51,
          300,    49, 13404,    18,   507,  3423,    72, 50118, 50118,   530,
          281,   991,    21,   416,    10,  1406,  9553,  1236, 16303,    11,
         1287,  1422,    77,    37,  1059,     5,  1482,     9,    22,  4310,
         3107,   843,   113,    11,  6200,     4, 50118, 50118,   133,  7504,
         5554,   311,     6,    61, 11590,   159,     5,   843,   144,  1406,
         3686,    11,     5,   315,   532,   716,    15, 18919,  4320,    18,
         6003,   727,   930,  5966,     6,   880,    15,    95,   707,  3188,
         4492,    53,  1335,  1059,    10,  1049, 20952,     9,  1583,     6,
           70,   198,     5,   232,     4, 50118, 50118,   133,   311,  4083,
         4356,   405,  3629,     6,   217,    10,  2384,  1732,     6,    22,
        12580,    18,  3107,   158,    60,    14,  1437, 10124,   991,  4457,
            4,     2])
tensor([    0, 27696,   377,    71,    39,   744,     6,  3188,  6718,  9239,
        10124,   991,    34,  1747,   303,    10, 18403,   317])
Six months after his death, radio personality Casey Kasem has finally found a resting place
tensor([50118,   133,   320,  1482,     9,    22,  4310,  3107,   843,   113,
           21,  8203,   719,   545,    23, 24534,   241, 38314, 23433,     6,
           10, 16233,    11, 26067,     6,  8683,     6,  1437,  1044,     9,
        16233,   942,  9795,  5387,  1469,   289,  2678,  8476,   174,  3480,
           15,   294])

The former host of "American Top 40" was buried December 16 at Vestre Gravlund, a cemetery in Oslo, Norway,  manager of cemetery administration Stein Olav Hohlen told CNN on Tuesday
tensor([50118,  1708,     5, 17353,    34,    45, 36771,    10,   251,    12,
        12364,   284, 13824])

But the burial has not calmed a long-running family feud
tensor([50118,   113,   713,   662,   127,   284,     8,    38,  2435,    14,
          127, 13404,    18, 14202,  1141,  5363,  4171, 10124,   991,     8,
           49,  1354,  7616,  2764,  7456,    10, 16233,    11,  8683,    88,
        34711,   127, 13404,    89,    60,    39,  1354,  9508,  1069, 10124,
          991,    26,    15,    69,   622,  1842])

"This morning my family and I learned that my Dad's abusive wife Jean Thompson Kasem and their daughter Liberty conned a cemetery in Norway into burying my Dad there," his daughter Kerri Kasem said on her Facebook page
tensor([50118,   113,  8170,    19, 12389,     5,  5430,  1666,    31,   127,
         1150,    18,   964,     8,   284,  9052,    14,    37,   770,     7,
           28,  8203,    11,     5, 27925, 37804,     6,     5,   247,    11,
           61,    37,    21,  2421,     8,  1179,     6,    39,  8605,    58,
            6,   683,   456,     6,  8266,    30,    39,  9515, 22724,  2650,
         1141])

"Even with ALL the letters ... from my father's friends and family stating that he wanted to be buried in the UNITED STATES, the country in which he was born and raised, his wishes were, once again, ignored by his unfaithful wife
tensor([50118, 35689, 10124,   991,   115,    45,    28,  1348,    13,  1129])

Jean Kasem could not be reached for comment
tensor([50118,   530,   281,   991,    18,  4194,   408,    31,    39,    78,
         3397,     8,    39,  1141,     9,    55,    87,   389,   107, 12248,
           11,   461,    11,     5,   377,   137, 10124,   991,   962,   502,
          379,    23,  1046,  7383])

Kasem's adult children from his first marriage and his wife of more than 30 years battled in court in the months before Kasem died June 15 at age 82
tensor([50118,   530,   254,  1069, 10124,   991,    21,   576,    10, 23923,
         3629,  4128,    81,    69,  1150,    18,   474,     6,    61,  5363,
        10124,   991,  8960,  4951,    11,   461,     6,  4564,    39,   408,
           58, 30088,  3558,    49,  1150,    18,   301])

Kerri Kasem was given a conservatorship over her father's health, which Jean Kasem subsequently fought in court, claiming his children were prematurely ending their father's life
tensor([50118, 17206, 10124,   991,    18,   744,     6,  5363, 10124,   991,
         2928,    69,    10,  7022,  1623,    31,    10,   575,  2122,    11,
         2005, 12811,     6,   886,     6,     8,   362,   123,     7,  1095,
           23,     5,   184,     9,   284,   964,    11,   663,   194])

Before Kasem's death, Jean Kasem removed her ailing husband from a care facility in Santa Monica, California, and took him to stay at the home of family friends in Washington state
tensor([50118,  4993, 10124,   991,   962,     6,   284,   453,  7311,    51,
          399,    75,   216,   147,     5,   809,    21,  2034])

After Kasem died, family members complained they didn't know where the body was located
tensor([50118, 36525,  6113,  8650,     6,     5,   285,   661,    13,  9508,
         1069, 10124,   991,     6,    26,    94,  1035,    14,  5363, 10124,
          991,  2928,    69,  1623,    18,   809,    31,    10,  6172,   184,
           11,   663])

Danny Deraney, the publicist for Kerri Kasem, said last summer that Jean Kasem removed her husband's body from a funeral home in Washington
tensor([50118,  4148,   302,     6,    37,    26,     6,    22, 30770,     6,
            5,   284,    16,    45,  1372,     8,    51,  1346,    14,  5363,
           64, 24569,   123, 11263,    79,  1072,     6,   190,   114,    24,
           16,    45,    99,  9239,   770])

On Monday, he said, "Clearly, the family is not happy and they understand that Jean can bury him wherever she wants, even if it is not what Casey wanted
tensor([50118,  1708,     5,  1159,  1437,    32,    23,  1987,  4730,    14,
           51,   300,    49, 13404,    18,   507,  3423])

But the kids  are at peace knowing that they got their Dad's final moments
tensor([50118,   530,   281,   991,    21,   416,    10,  1406,  9553,  1236,
        16303,    11,  1287,  1422,    77,    37,  1059,     5,  1482,     9,
           22,  4310,  3107,   843,   113,    11,  6200])

Kasem was already a popular disc jockey in Los Angeles when he became the host of "American Top 40" in 1970
tensor([50118,   133,  7504,  5554,   311,     6,    61, 11590,   159,     5,
          843,   144,  1406,  3686,    11,     5,   315,   532,   716,    15,
        18919,  4320,    18,  6003,   727,   930,  5966,     6,   880,    15,
           95,   707,  3188,  4492,    53,  1335,  1059,    10,  1049, 20952,
            9,  1583,     6,    70,   198,     5,   232])

The syndicated show, which counted down the 40 most popular songs in the United States based on Billboard magazine's Hot 100 music chart, began on just seven radio stations but quickly became a mainstay of thousands, all around the world
tensor([50118,   133,   311,  4083,  4356,   405,  3629,     6,   217,    10,
         2384,  1732,     6,    22, 12580,    18,  3107,   158,    60,    14,
         1437, 10124,   991,  4457])

'''

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
    simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
    simcse_model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large").cuda()
    bart_model = torch.hub.load('pytorch/fairseq', 'bart.large') # used for tokenize
    bart_tokenize = (bart_model.encode, bart_model.decode)

    output_dir = 'data/cnn'
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    split_size = {
        'train': 10000,
        'valid': 5000,
        'test': 1000
    }

    for split in ['train', 'valid', 'test']:
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

            tgt_relevance = compute_tgt_relevance(simcse_model, simcse_tokenizer, bart_tokenize, src, tgt)

            examples.append({
                'condition': src,
                'text': tgt,
                'tgt_relevance' : tgt_relevance
            })

            if len(examples) >= split_size[split]:
                break

        print(f'#{split}: {len(examples)}')
        pickle.dump(examples, open(f'{output_dir}/{split}.pickle', 'wb'))

if __name__ == "__main__":
    cnn()
    # wp()