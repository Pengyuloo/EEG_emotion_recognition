# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from EncoderRNN import EncoderRNN

class lstmCapsule(object):
    def __init__(self,
            dim_input,
            dim_hidden,
            channels, 
            n_layers,
            n_label,
            batch_size,
            learning_rate,
            weight_decay=0,
            cell_dropout_rate=0.,
            final_dropout_rate=0.,
            bidirectional=True,
            optim_type="Adam",
            rnn_type="LSTM",
            use_cuda=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model = EncoderRNN(dim_input, dim_hidden, channels, n_layers, n_label, cell_dropout_rate, final_dropout_rate,  rnn_type, bidirectional, use_cuda)
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = getattr(optim, optim_type)([
                                        {'params': self.model.base_params}
                                    ], lr=self.learning_rate, weight_decay=weight_decay)
        self.encoder_hidden = self.model.init_hidden(self.batch_size) #20190930 add init_hidden
        
    def init(self):
        self.model.weights_init()


    def stepTrain(self, batched_data, batched_label, inference=False):
        self.model.eval() if inference else self.model.train()
        input_variable = Variable(torch.from_numpy(batched_data).float())
        input_variable = input_variable.cuda() if self.use_cuda else input_variable
        tensor_label = Variable(torch.from_numpy(batched_label).long())
        tensor_label = tensor_label.cuda() if self.use_cuda else tensor_label
        hidden = self.model.init_hidden(input_variable.shape[0])

        if inference == False:
            self.optimizer.zero_grad()
            
        prob = self.model(input_variable, hidden, tensor_label)
        # print(prob)
        softmaxloss = torch.nn.CrossEntropyLoss()
        loss = softmaxloss(prob, tensor_label)
        print("current loss: ",loss.item())
        
        if inference == False:
            L1_reg = 0
            for param in self.model.parameters():
                L1_reg += torch.sum(torch.abs(param))
            loss += 0.001 * L1_reg  # lambda=0.001
            
            loss.backward()
            self.optimizer.step()

        return np.array(loss.data.cpu().numpy()).reshape(1), prob.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self, '%s/model%s.pkl' % (dir, idx))