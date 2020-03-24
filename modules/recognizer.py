#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from base.base_model import BaseModel 

class BidirectionaLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionaLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden*2, n_out)
        
    def forward(self, input, lengths):
        """
        :return: A tensor with shape N * W * nOut
        """
        
        self.rnn.flatten_parameters()
        total_length = input.size(1) # W
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True) # pack padded input
        recurrent, _ = self.rnn(packed_input)  # [T, b, h * 2] [W, N, nh*2]
        padded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(recurrent, total_length=total_length, batch_first=True)

        b, T, h = padded_input.size() # [N, W, nh * 2]
        t_rec = padded_input.contiguous().view(T * b, h) # [W * N, nh * 2]
        output = self.fc(t_rec)  # [T * b, nOut] [W * N, nOut]
        output = output.view(b, T, -1) # [N, W, nOut]
        output = nn.functional.log_softmax(output, dim=-1) # required by pytorch's ctcloss
        return output

class Recognizer(BaseModel):
    def __init__(self, n_class, nc, nh, config):
        super().__init__(config)
        ks = [3, 3, 3, 3, 3, 3]
        ss = [1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256]
        self.cnn = nn.Sequential()
        def conv_bn_relu(i):
            nIn = nc if i == 0 else nm[i-1]
            nOut = nm[i]
            self.cnn.add_module('conv{}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            self.cnn.add_module('batchnorm{}'.format(i), nn.BatchNorm2d(nOut))
            self.cnn.add_module('relu{}'.format(i), nn.ReLU())
        
        def height_max_pool(i):
            self.cnn.add_module('height-max-pool{}'.format(i), nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)))
        
        conv_bn_relu(0) # N * 64 * H * W
        conv_bn_relu(1) # N * 64 * H * W
        height_max_pool(0) # N * 64 * 4 * W 
        conv_bn_relu(2) # N * 128 * 4 * W
        conv_bn_relu(3) # N * 128 * 4 * W
        height_max_pool(1) # N * 128 * 2 * W
        conv_bn_relu(4) # N * 256 * 2 * W
        conv_bn_relu(5) # N * 256 * 2 * W
        height_max_pool(2) # N * 256 * 1 * W
        
        self.rnn = BidirectionaLSTM(256, nh, n_class)
        
    def forward(self, input, lengths):
        """
        :param input: N * C * H(=8) * W (tensor)
        :param lengths: N
        """
        conv = self.cnn(input) # N * 256 * 1 * W
        conv = conv.squeeze(2) # N * 256 * W
        conv = conv.permute(0, 2, 1)  # [B, T, C] (N * W * 256)
        # rnn features
        output = self.rnn(conv, lengths)
        return output


class RecognizerLoss(nn.Module):
    def __init__(self):
        super(RecognizerLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss()
        
    def forward(self, gt, pred):
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        return loss

