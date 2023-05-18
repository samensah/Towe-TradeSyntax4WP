"""
models for sentiment.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ON_LSTM import ONLSTMStack
from tree import *


from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

# 0: tokens, 1: position, 2: POS, 3: head, 4: deprel, 5: selfloop, 6: mask_s,  7: label
class Toy_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        conf = BertConfig.from_pretrained(args.bert_model)
        self.encoder = BertModel.from_pretrained(args.bert_model, config=conf)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True 
        
        self.position_fuse = ONLSTMStack([args.dim_bert+args.dim_position, 2*args.dim_bilstm_hidden], chunk_size=args.chunk_size)
        if args.dim_position != 0:
            self.position_emb = nn.Embedding(len(args.position2id), args.dim_position)

        # classifier
        self.classifier   = nn.Linear(2*args.dim_bilstm_hidden, len(args.bio2id))

        # loss function 
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')

    # [words, position, mask_s, label]
    def forward(self, inputs):
        
        tokens, position, mask_s, label  = inputs
        lens = mask_s.sum(dim=1)

        # Bert Encoder
        token_idx, segment_idx, mask, mask_s_padded = to_bert_input(tokens, 0, mask_s)
        H, _ = self.encoder(token_idx, segment_idx, mask)
        
        # remove aspect tail
        if self.args.use_A == 1:
            H = H * mask_s_padded.unsqueeze(-1)
            H = H[:,:int(lens[0].item()),:]

        if self.args.dim_position != 0:
            H = torch.cat([H, self.position_emb(position)], dim=-1)

        # ON LSTM to fuse the postion information
        H = self.position_fuse(H.transpose(0,1), self.position_fuse.init_hidden(len(tokens)))[0]
        H = H.transpose(0,1)
        
        # use H for classification
        logits = self.classifier(H)

        # pred and loss
        opn_pred = torch.argmax(logits, dim=2)
        opn_pred = opn_pred * mask_s.long()
        opn_loss = self.ce_loss(logits.reshape(-1, logits.size(-1)), label.long().reshape(-1))
        opn_loss = opn_loss.reshape(logits.size(0), logits.size(1))
        opn_loss = (opn_loss * mask_s).sum() / opn_loss.size(0)

        return opn_loss, opn_pred.tolist()

# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.dim_bert + args.dim_position
        self.rnn = nn.LSTM(self.in_dim, args.dim_bilstm_hidden, 1, batch_first=True, dropout=0.0, bidirectional=True)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.dim_bilstm_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        # unpack inputs
        inputs, lens = inputs[0], inputs[1]
        return self.encode_with_rnn(inputs, lens, inputs.size()[0])

# Initialize zero state
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


def to_bert_input(token_idx, null_idx, mask_real):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    if segment_idx.size(1) != mask_real.size(1):
        pad_t = torch.zeros(segment_idx.size(0), segment_idx.size(1)-mask_real.size(1)).cuda()
        mask_pad = torch.cat([mask_real, pad_t], dim=-1)
        segment_idx = torch.where(mask_pad==1, torch.zeros_like(segment_idx), torch.ones_like(segment_idx))
    else:
        mask_pad = None
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask, mask_pad
