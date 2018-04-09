# coding:utf8
import pandas as pd
import numpy as np
import pickle
import torch
from config import params
from sswe import SSWE

if __name__ == '__main__':
    
    model = SSWE(params)
    model.load_state_dict(torch.load('../docs/model/sswe_01-22-10:03'))
    lookup = model.lookup.weight.data.cpu().numpy()
    pickle.dump(lookup,open('../docs/model/lookup_01-22-10:03','wb'))
    exit()


    df = pd.read_pickle('../docs/data/HML_JD_ALL.new.dat')
    oj = (df['Event']>0) | (df['Agent']>0)|(df['Object']>0) 
    fj = (df['Satisfaction']>0)|(df['Disappointment']>0)|(df['Admiration']>0)|(df['Reproach']>0)|(df['Like']>0)|(df['Dislike']>0)
    print(df[oj&fj][:10].reset_index(drop=True))
    df=df[oj&fj].reset_index(drop=True)
    df['cv'] = np.random.randint(10,size=len(df))
    print(df.head())
    # exit()
    df.to_pickle('../docs/data/HML_data_clean.dat')
