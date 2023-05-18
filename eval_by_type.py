import os
import shutil
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle
import copy
import json

from config import *
from evals import *
from loader import DataLoader 
from trainer import MyTrainer
from pytorch_transformers.tokenization_bert import BertTokenizer

seed = random.randint(1, 10000)

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)


# load vocab and embedding matrix
dataset_path          = "./data/%s"        % (args.dataset)

# load data
train_path  = '%s/train.json' % (dataset_path)
test_path   = '%s/test.json'  % (dataset_path)

# generate POS2id, bio2id, position2id, rel2id
args.bio2id = {'O':0, 'B':1, 'I':2}
max_len = 512
position_list = []
for i in range(1, max_len+1, 1):
    position_list.append(i)
    position_list.append(-i)
position_list = ['[PAD]', 0] + position_list
args.position2id = {p:i for i,p in enumerate(position_list)}

args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print("Loading data from {} with batch size {}...".format(dataset_path, args.train_batch_size))
train_batches  = DataLoader(args, train_path, mode='train')
print("Loading data from {} with batch size {}...".format(dataset_path, args.eval_batch_size))
test_batches   = DataLoader(args, test_path,  mode='test')


print('Building model...')
# create model
trainer_model  = MyTrainer(args)
trainer_model.load(args.load_dir)
scores, predictions = evaluate_program_case_study(trainer_model, test_batches, args)
print(scores)
sample_id, golden, pred = predictions

data_dict = dict()
for d in test_batches.raw_data:
    data_dict[d['sample_id']] = [d['tokens'], d['asp']]

ret = [] 

for s_id, g, p in zip(sample_id, golden, pred):
    tokens_ori, asp = data_dict[int(s_id)]

    tokens_wp = [] 
    for t in tokens_ori:
        tokens_wp.extend(args.tokenizer.tokenize(t))
    tokens_wp = ['[CLS]']+tokens_wp+['[SEP]']

    g_opn = [tokens_wp[gg[0]:gg[1]+1] for gg in g]
    p_opn = [tokens_wp[pp[0]:pp[1]+1] for pp in p]
    s, e = asp[0][0], asp[0][1]

    ret.append({'tokens': tokens_wp, 'aspect_term': tokens_ori[s:e+1], 'golden':g_opn, 'prediction': p_opn})

with open('pair_'+args.dataset+'.json', 'w') as ouf:
    json.dump(ret, ouf, indent=4)



