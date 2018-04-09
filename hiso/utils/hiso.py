#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/6 11:13
# @From    : PyCharm
# @File    : hiso
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from gensim.models import Word2Vec

class HISO(nn.Module):
    def __init__(self, opt):
        super(HISO, self).__init__()

        self.model_name = 'HISO'
        self.opt = opt
        # Embedding Layer
        self.wd_embed = nn.Embedding(opt.voc_size, opt.embed_dim)
        self.pos_embed = nn.Embedding(opt.pos_size, opt.embed_dim)
        self.initEmbedWeight()
        # conv layer
        self.word_conv = self.flatConv
        self.pos_conv = self.flatConv
        # Bi-GRU Layer
        self.wd_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True,
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )
        self.word_squish_w = nn.Parameter(torch.randn(2*opt.ghid_size, 2*opt.ghid_size))
        self.word_atten_proj = nn.Parameter(torch.randn(2*opt.ghid_size, 1))

        # Bi-GRU Layer
        self.pos_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True, 
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )
        self.pos_squish_w = nn.Parameter(torch.randn(2*opt.ghid_size, 2*opt.ghid_size))
        self.pos_atten_proj = nn.Parameter(torch.randn(2*opt.ghid_size, 1))

        # output from pos hidden layer to predict middle labels
        pos_hidden_size = opt.ghid_size * opt.glayer
        self.pos_fc = nn.Sequential(
                nn.BatchNorm1d(pos_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(pos_hidden_size, opt.auxiliary_labels),
                nn.Softmax(dim=-1)
                )
        # predict final labels
        combine_size = opt.ghid_size * opt.glayer + opt.auxiliary_labels
        self.fc = nn.Sequential(
                nn.Linear(combine_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),
                nn.Linear(128, opt.label_dim),
                nn.Softmax(dim=-1)
                )
        self.softmax = nn.Softmax(dim=1)

    def deepConv(self, x):
        '''
        stackig conv layer 
        '''
        x = torch.transpose(x, 1, 2)
        kernel_size = [1, 2, 2]
        filter_num = [self.opt.embed_dim, 128, 128, 100]
        for i, k_s in enumerate(kernel_size):
            x = nn.Conv1d(in_channels = filter_num[i],
                    out_channels = filter_num[i+1],
                    kernel_size = kernel_size[i],
                    stride = 1)(x)
        return torch.transpose(x, 1, 2)

    def flatConv(self, x):
        '''
        Flatting feature after conv parallel
        '''
        # embed_dim[N, L, C] --> [N, C, L]
        x = torch.transpose(x, 1, 2)
        kernel_size = [1, 2]
        filter_num = [50, 50, 48]
        output, min_len = [], x.size()[-1]
        for i, k_s in enumerate(kernel_size):
            cur_res = nn.Conv1d(in_channels = x.size()[1],
                    out_channels = filter_num[i],
                    kernel_size = kernel_size[i],
                    stride = 1,
                    padding=int(kernel_size[i]/2))(x)

            min_len = min(min_len, cur_res.size()[-1])
            output.append(cur_res)
        result = torch.cat([y[:,:,:min_len] for y in output], 1)

        return torch.transpose(result, 1, 2)

        

    def initEmbedWeight(self):
        '''
        init embedding layer from random|word2vec|sswe
        '''
        if 'w2v' in self.opt.init_embed:
            weights = Word2Vec.load('../docs/data/w2v_word_100d_5win_5min')
            voc = json.load(open('../docs/data/voc.json','r'))['voc']
            print(weights[list(voc.keys())[3]])

            word_weight = np.zeros((len(voc),self.opt.embed_dim))
            for wd,idx in voc.items():
                vec = weights[wd] if wd in weights else np.random.randn(self.opt.embed_dim)
                word_weight[idx] = vec
            # print(word_weight[3])
            self.wd_embed.weight.data.copy_(torch.from_numpy(word_weight))

            weights = Word2Vec.load('../docs/data/w2v_pos_100d_5win_5min')
            pos = json.load(open('../docs/data/pos.json','r'))['voc']
            pos_weight = np.zeros((len(pos),self.opt.embed_dim))
            for ps,idx in pos.items():
                vec = weights[ps] if ps in weights else np.random.randn(self.opt.embed_dim)
                pos_weight[idx] = vec
            self.pos_embed.weight.data.copy_(torch.from_numpy(pos_weight))

        elif 'sswe' in self.opt.init_embed:
            word_weight = pickle.load(open('../docs/model/%s'% self.opt.embed_path,'rb'))
            self.wd_embed.weight.data.copy_(torch.from_numpy(word_weight))
        # random default


    def forward(self, wd, pos):
        # encoder
        wd = self.wd_embed(wd)
        pos = self.pos_embed(pos)
        #print(pos.size())
        # conv layer
        wd  = self.word_conv(wd)
        pos = self.pos_conv(pos)
        #print(wd.size())

        # Bi-GRU
        h0 = self.init_hidden(wd.size()[0])
        wd_out, wd_hidden = self.wd_bi_gru(wd)
        pos_out, pos_hidden = self.pos_bi_gru(pos)
        # attention
        if 'word' in self.opt.attention:
            wd_atten = self.attention(wd_out,weight_input=wd_out)
        elif 'pos' in self.opt.attention:
            wd_atten = self.attention(wd_out,weight_input=pos_out)
        else:
            wd_atten = wd_out[:,-1,:]
        
        # pos_out to predict auxiliary label
        auxi_probs = self.pos_fc(pos_out[:, -1, :])

        # combine wd_out with auxi_probs as feature
        combine_feature = torch.cat((wd_atten, auxi_probs), dim=1)
        logits = self.fc(combine_feature)

        return logits, auxi_probs

    def attention(self,seq,weight_input):
        '''
        attention layer
        '''
        _squish = batch_matmul(weight_input, self.word_squish_w, active_func='tanh')
        att_weight = batch_matmul(_squish, self.word_atten_proj)
        att_weight_norm = self.softmax(att_weight)
        _out = attention_matmul(seq, att_weight_norm)

        return _out


    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.opt.glayer, batch_size, self.opt.ghid_size)
        return Variable(h0)


def batch_matmul(seq,weight,active_func=''):
    '''
    seq matmul weight with keeping dims
    '''
    out = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if active_func == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)

        out = _s if out is None else torch.cat((out, _s), 0)
    return out

def attention_matmul(seq, att_weight):
    '''
    apply att_weight on seq
    '''
    att_out = None
    for i in range(seq.size(0)):
        s_i = seq[i] * att_weight[i]
        s_i = s_i.unsqueeze(0)
        
        att_out = s_i if att_out is None else torch.cat((att_out,s_i),0)
    return torch.sum(att_out,dim=1)


class HisoLoss(nn.Module):
    def __init__(self, opt):
        super(HisoLoss, self).__init__()
        self.opt = opt
        # self.reconstruction_loss = 0 // todo

    def forward(self,auxi_probs, auxi_labels, final_probs, final_labels):
        # calcu auxi_labels margin loss
        self.auxi_loss = self.marginLoss(auxi_probs, auxi_labels)

        # calcu final_labels margin loss
        self.final_loss = self.marginLoss(final_probs, final_labels)
        
        self.loss = self.opt.loss_alpha * self.auxi_loss + self.final_loss
        return self.loss

    def marginLoss(self, probs, labels):
        
        left = F.relu(self.opt.max_margin - probs, inplace=True)**2
        right = F.relu(self.opt.min_margin - probs, inplace=True)**2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        return margin_loss.sum() / labels.size(0)


class opt(object):
    voc_size = 23757
    pos_size = 57
    embed_dim = 100
    ghid_size = 3
    seq_len = 4
    glayer = 2
    auxiliary_labels = 3
    label_dim = 6
    max_margin = 0.9
    min_margin = 0.1
    embed_path='lookup_01-22-19:10'
    init_embed='randn'
    loss_alpha=1e-2
    attention='word'

if __name__ == '__main__':
    import visdom
    from visualize import Visualizer   
    # vis = Visualizer('main_test',port=8099)
    import time
    import torch.optim as optim
    wd = Variable(torch.LongTensor([[2,45,75,34], [5,54,76,23]]))
    pos = Variable(torch.LongTensor([[3,45,8,2], [13,56,7,43]]))
    labels = Variable(torch.FloatTensor([[1,0,0,1,0,0],[0,0,1,0,1,0]]))
    auxi = Variable(torch.FloatTensor([[1,0,0],[0,1,0]]))

    model = HISO(opt)
    Loss = HisoLoss(opt)
    op = optim.SGD(model.parameters(),lr=0.1)
    model.train()
    a,b= model.word_squish_w.data,model.word_atten_proj.data
    print(a,b)
    for i in range(100):
        final_probs,auxi_probs = model(wd, pos)
        loss = Loss(auxi_probs, auxi, final_probs, labels)
        op.zero_grad()
        loss.backward()
        op.step()
        print(loss.data[0])
    #    print(final_probs.data,auxi_probs.data)
    model.eval()
    fp,ap = model(wd,pos)
    print(fp.data,ap.data)
    print(model.word_atten_proj.data)
    print(model.word_squish_w.data)
        #vis.plot('loss', loss.data[0])
        #time.sleep(1)
        #vis.log({'epoch':i,'loss':loss.data[0]})
    # print(outputs)
