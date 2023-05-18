import json
import random
import torch
import numpy as np

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, args, file_path, mode='train'):
        self.args = args
        self.file_path = file_path
        self.tokenizer = args.tokenizer
        if mode == 'train':
            self.batch_size = args.train_batch_size
        elif mode == 'test':
            self.batch_size = args.eval_batch_size

        with open(file_path, 'r') as f:
            self.raw_data = json.load(f)

        # generate sample id
        count = 0
        for d in self.raw_data:
            d['sample_id'] = count
            count += 1

        self.position2id, self.bio2id = args.position2id, args.bio2id
        self.data = self.preprocess(self.raw_data)
        self.num_examples = len(self.data)

        # chunk into batches
        self.data = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        print("{} batches created for {}".format(len(self.data), self.file_path))

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []

        for d in data:
            # dict_keys(['tokens', 'POS', 'deprel', 'asp', 'opn'])
            tokens   = d['tokens']

            token_index_map = {}
            tokens_wp = []
            for i in range(len(tokens)):
                t  = tokens[i]
                pt = self.tokenizer.tokenize(t)
                token_index_map[i] = len(tokens_wp)
                if self.args.use_wordpiece == 1:
                    tokens_wp.extend(pt)
                else:
                    tokens_wp.append(t)

            tokens_wp = ['[CLS]'] + tokens_wp + ['[SEP]']
            for key in token_index_map.keys():
                token_index_map[key] = token_index_map[key] + 1
            real_length = len(tokens_wp)

            asp = d['asp'][0]
            asp_from = token_index_map[asp[0]]
            if asp[1] == len(tokens) - 1:
                asp_to = real_length - 1 - 1 
            else:
                asp_to = token_index_map[asp[1]+1]-1

            if self.args.use_A == 1:
                tokens_wp = tokens_wp + tokens_wp[asp_from:asp_to+1] + ['[SEP]']
            
            # map to ids
            tokens_id = self.tokenizer.convert_tokens_to_ids(tokens_wp)

            if self.args.use_mask == 1:
                # set aspect words in the sentence to [PAD]
                tokens_id[asp_from:asp_to+1] = [0]*(asp_to-asp_from+1)

            label = ['O' for _ in range(real_length)]
            for opn in d['opn']:
                opn_sta = token_index_map[opn[0]]
                if opn[1] == len(tokens) - 1:
                    opn_end = real_length - 1 - 1 
                else:
                    opn_end = token_index_map[opn[1]+1]-1

                if opn_sta == opn_end:
                    label[opn_sta] = 'B'
                else:
                    label[opn_sta] = 'B'
                    for i in range(opn_sta+1, opn_end+1, 1):
                        label[i] = 'I'
            label = map_to_ids(label, self.bio2id)
            
            position = [i-asp_from for i in range(asp_from)] + [0 for _ in range(asp_from, asp_to+1, 1)] + [i-asp_to for i in range(asp_to+1, real_length, 1)]
            position = map_to_ids(position, self.position2id) 

            assert len(position) == len(label) == real_length

            mask_s = [1 for i in range(real_length)]

            processed.append([d['sample_id'], tokens_id, position, mask_s, label])

        return processed

    def __len__(self):
        return len(self.data)

    # 0: sample_id, 1: tokens, 2: position, 3: mask_s, 4: label
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort all fields by lens for easy RNN operations
        lens = [sum(x) for x in batch[3]]
        batch, _ = sort_all(batch, lens)

        # convert to tensors
        words       = get_long_tensor(batch[1], batch_size)
        position    = get_long_tensor(batch[2], batch_size)
        mask_s      = get_float_tensor(batch[3], batch_size)
        label       = get_float_tensor(batch[4], batch_size)
                
        return [batch[0], words, position, mask_s, label] 

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else vocab['[UNK]'] for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

