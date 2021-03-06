#coding:utf-8

import pickle
import torch
import torch.nn as nn
import pandas as pd
import os,json,time
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.visualize import Visualizer
from gensim.models import Word2Vec
import os
import sys
from tqdm import tqdm
# 设置gpu
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
experimental_msg = 'W2V_NULL_alpha07'
vis = Visualizer(env='SSWE',port=8099,log_dir='myruns/sswe_%s_%s'%(time.strftime('%m-%d-%H-%M',time.localtime()),experimental_msg))


class Score(nn.Module):
    '''
    全连接层，计算gram语义，情感语义得分
    '''
    def __init__(self,opt):
        super(Score, self).__init__()
        
        self.synt_fc = nn.Sequential(
                nn.Linear(3*opt.embed_dim,128),
                nn.Hardtanh(inplace=True),
                nn.Linear(128,1)
                )

        self.sent_fc = nn.Sequential(
                nn.Linear(3*opt.embed_dim,128),
                nn.Hardtanh(inplace=True),
                nn.Linear(128,2),
                nn.Softmax(dim=-1)
                )

    def forward(self,x):
        synt_score = self.synt_fc(x)
        sent_score = self.sent_fc(x)

        return synt_score, sent_score

class SSWE(nn.Module):
    '''
    SSWE 情感语义模型
    '''
    def __init__(self, opt):
        super(SSWE, self).__init__()
        
        self.opt = opt
        self.lookup = nn.Embedding(opt.voc_size,opt.embed_dim)
        # support init embeding weight using Word2Vec
        self.initEmbedWeight()
        self.score = Score(opt)

    def forward(self,x):
        '''
        x = [left,right, middle, corrput, corrput]
        '''
        x = self.lookup(x)
        true_gram = torch.cat((x[:,0,:],x[:,1,:],x[:,2,:]),dim=1)
        scores = [self.score(true_gram)]

        for step in range(3, x.size(1)):
            corrupt_gram = torch.cat((x[:,0,:],x[:,1,:],x[:,step,:]),dim=1)

            scores.append(self.score(corrupt_gram))
        return scores

    def initEmbedWeight(self):
        '''
        init embeding weight using Word2Vec|rand
        '''
        if 'w2v' in self.opt.init_sswe_embed:
            weights = Word2Vec.load('%s/docs/data/w2v_word_100d_5win_5min'%BASE_DIR)
            voc = json.load(open('%s/docs/data/voc.json'%BASE_DIR,'r'))['voc']

            word_weight = np.zeros((self.opt.voc_size, self.opt.embed_dim))
            for wd,idx in voc.items():
                vec = weights[wd] if wd in weights else np.random.randn(self.opt.embed_dim)
                word_weight[idx] = vec
            # print(word_weight[3])
            self.lookup.weight.data.copy_(torch.from_numpy(word_weight))

class HingeMarginLoss(nn.Module):
    '''
    计算hinge loss 接口
    '''
    def __init__(self):
        super(HingeMarginLoss, self).__init__()

    def forward(self,t,tr,delt=None,size_average=False):
        '''
        计算hingle loss
        '''
        if delt is None:
            loss = torch.clamp(1-t+tr, min=0)
        else:
            loss = torch.clamp(1 - torch.mul(t-tr, torch.squeeze(delt)), min=0)
            loss = torch.unsqueeze(loss,dim=-1)
        # loss = torch.mean(loss) if size_average else torch.sum(loss) 
        return loss

class SSWELoss(nn.Module):
    '''
    SSWE 中考虑gram得分和情感得分的loss类
    '''
    def __init__(self, alpha=0.5):
        super(SSWELoss,self).__init__()
        
        # loss weight for trade-off
        self.alpha = alpha
        self.hingeLoss = HingeMarginLoss()

    def forward(self,scores, labels, size_average=False):
        '''
        [(true_sy_score,true_sem_score), (corrput_sy_score,corrupt_sem_score),...]
        '''
        assert len(scores) >= 2
        true_score = scores[0]
        # all gram have same sentiment hingle loss, because don't use corrupt gram
        sem_loss = self.hingeLoss(true_score[1][:,0],true_score[1][:,1],delt=labels)
        loss = []

        for corpt_i in range(1, len(scores)):
            syn_loss = self.hingeLoss(true_score[0], scores[corpt_i][0])
            cur_loss = syn_loss * self.alpha + sem_loss * (1 - self.alpha )

            loss.append(cur_loss)

        loss = torch.cat(loss,dim=0)

        if size_average:
            loss = torch.mean(loss)

        return loss


class SemDataSet(Dataset):
    def __init__(self, file_path,voc_path,pos_path):
        
        self.df = pd.read_pickle(file_path)

        # load directly if voc file exists.
        if os.path.isfile(voc_path) and os.path.isfile(pos_path):
            with open(voc_path, 'r') as fv, open(pos_path, 'r') as fp:
                voc_dict = json.load(fv)
                self.voc = voc_dict['voc']
                max_length = voc_dict['max_length']

                pos_dict = json.load(fp)
                self.pos = pos_dict['voc']
        else:
            self.voc, self.pos, max_length = build_vocab(file_path, voc_path, pos_path)
        # 将词处理成id
        self.all_sen_wd = [[self.voc.get(d[0],0) for d in wd_pos] for wd_pos in self.df['Cut']]
        # self.all_sen_wd = list(filter(lambda wd_ids:len(wd_ids)>2, all_sen_wd))
        self.voc_len = len(self.voc)
        self.deal()

        self.floatTensor = torch.FloatTensor
        self.longTensor = torch.LongTensor

    def __len__(self):

        return len(self.x_y)

    def __getitem__(self,idx):
        sample = {
                'gram':self.longTensor(self.x_y[idx]['gram']),
                'label':self.floatTensor(self.x_y[idx]['label'])
                }
        return sample

    def deal(self):
        '''
        先处理完所有数据，方便计数
        '''
        self.x_y = []
        for i, sent_wds in enumerate(self.all_sen_wd):
            # 过滤词长小于3的
            if len(sent_wds) < 3: continue
            # labels: 满意|欣赏|喜欢-->1  失望|责备|不喜欢--> -1
            label = 1 if self.df['Satisfaction'][i] or self.df['Admiration'][i] or self.df['Like'][i] else -1
            for wd_i in range(1,len(sent_wds)-1):
                self.x_y.append({
                    'gram':sent_wds[wd_i-1:wd_i+2]+np.random.randint(self.voc_len,size=5).tolist(),
                    'label':[label]
                    })

def trainSSWE():
    from config import params
    use_cuda = torch.cuda.is_available()
    # syn_alpha = 0.7

    sem_data = SemDataSet('%s/docs/data/HML_data_clean.dat'%BASE_DIR,
            voc_path='%s/docs/data/voc.json'%BASE_DIR,
            pos_path='%s/docs/data/pos.json'%BASE_DIR)
    loader = DataLoader(sem_data,shuffle=True,batch_size=64)

    sswe = SSWE(params)
    s_loss = SSWELoss(alpha=params.sswe_alpha)
    if use_cuda:
        sswe.cuda()
        s_loss.cuda()

    optimizer = optim.SGD(sswe.parameters(),lr=0.1,momentum=0.9)
    scheduler = MultiStepLR(optimizer,milestones=[int(0.3*params.epochs),int(0.7*params.epochs)],gamma=0.1)

    total_loss = []
    for epoch in tqdm(range(params.epochs)):
        scheduler.step()
        for batch_idx, samples in enumerate(loader,0):
            v_gram = Variable(samples['gram'].cuda() if use_cuda else samples['gram'])
            v_label = Variable(samples['label'].cuda() if use_cuda else samples['label'])

            scores = sswe(v_gram)
            optimizer.zero_grad()
            loss = s_loss(scores, v_label, size_average=True)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.data[0])

            if batch_idx % 20 == 1:
                vis.plot('Hinge loss',np.mean(total_loss))
                print
    # save model
    timestamp = time.strftime('%m-%d-%H:%M',time.localtime())
    torch.save(sswe.state_dict(),'%s/docs/model/sswe_alpha%s_%s'%(BASE_DIR, params.sswe_alpha, timestamp))
    lookup = sswe.lookup.weight.data.cpu().numpy() 
    pickle.dump(lookup, open('%s/docs/model/lookup_alpha%s_%s'%(BASE_DIR, params.sswe_alpha, timestamp),'wb'))
    
if __name__ == '__main__':
    trainSSWE()
    exit()

    # x = Variable(torch.LongTensor([[2,45,34,56,94,3,35,32], [25,76,18,23,56,23,56,64]]))
    # y = Variable(torch.FloatTensor([[1], [-1]]))
    sswe = SSWE(opt)
    s_loss = SSWELoss()
    scores = sswe(x)
    loss = s_loss(scores, y,size_average=False)
    print(loss.data)
