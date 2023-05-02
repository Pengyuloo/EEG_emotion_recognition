# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self,
            dim_input,  # 320
            dim_hidden,  # 256
            channels,
            n_layers,
            n_label,
            cell_dropout_rate,
            final_dropout_rate,
            rnn_type,
            bidirectional,
            use_cuda):
        super(EncoderRNN, self).__init__()
        self.dim_input = dim_input
        self.channels = channels
        self.n_label = n_label
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.rnn_type =  rnn_type
        self.bidirectional = bidirectional
        self.dim_hidden = dim_hidden
        
        self.add_module('rnn', getattr(nn, self.rnn_type)(self.dim_input, self.dim_hidden, self.n_layers, batch_first=True, bidirectional=bidirectional,dropout=cell_dropout_rate))

        self.add_module('fc', nn.Linear(dim_hidden*channels, n_label))

        self.base_params = self.parameters()

    def forward(self, input, hidden, tensor_label):
        
        
        output, hidden = self.rnn(input,hidden)
        output = torch.reshape(output, [-1, self.dim_hidden*self.channels])
        fc_output = self.fc(output)
        

        return fc_output

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.n_layers * (2 if self.bidirectional else 1), batch_size, self.dim_hidden).zero_(), requires_grad=False)
        h_0 = h_0.cuda() if self.use_cuda else h_0
        return (h_0, h_0) if self.rnn_type == "LSTM" else h_0
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.RNN):
                #xavier_normal 高斯分布
                #kaiming_normal_高斯分布
                #kaiming_uniform_均匀分布
                torch.nn.init.kaiming_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu')) #gain=1.0
                torch.nn.init.kaiming_uniform_(m.bias.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
