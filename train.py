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
import time 
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

# create the folder for saving the best model
args.save_dir = args.save_dir+'_'+args.dataset
if os.path.exists(args.save_dir) != True:
    os.mkdir(args.save_dir)

log_file = FileLogger(args.save_dir+"/log.txt")

print('Building model...')
# create model
trainer_model  = MyTrainer(args)

# start training
estop      = 0
batch_num  = len(train_batches)
current_best_F1 = -1
for epoch in range(1, args.n_epoch+1):
    
    if estop > args.early_stop:
        break
    st = time.time()

    train_loss, train_step = 0., 0
    for i in range(batch_num):
        batch = train_batches[i]
        loss = trainer_model.update(batch)
        train_loss += loss
        train_step += 1
        
        # print training loss 
        if train_step % args.print_step == 0:
            print("[{}] train_loss: {:.4f}".format(epoch, train_loss/train_step))

    et = time.time()
    print(et-st)
    
    # evaluate on unlabel set
    print("")
    print("Evaluating...Epoch: {}".format(epoch))
    eval_scores, eval_loss = evaluate_program(trainer_model, test_batches, args)
    print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
    # loging
    log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

    if eval_scores[-1] > current_best_F1:
        current_best_F1 = eval_scores[-1]
        trainer_model.save(args.save_dir+'/best_model.pt')
        print("New best model saved!")
        log_file.log("New best model saved!")
        estop = 0

    estop += 1
    print("")


print("Training ended with {} epochs.".format(epoch))

# final results
trainer_model.load(args.save_dir+'/best_model.pt')
eval_scores, eval_loss = evaluate_program(trainer_model, test_batches, args)

print("Final result:")
print("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))

# loging
log_file.log("Final result:")
log_file.log("Prec: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_scores[0], eval_scores[1], eval_scores[2]))
