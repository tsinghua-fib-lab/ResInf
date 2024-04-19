import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import random
from time import time
from collections import defaultdict
import pickle
import warnings
# from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
from os import error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import torch
import json

class NetDataset2(Dataset):
    
    def __init__(self, mode, type, args):
        super(NetDataset2, self).__init__()
        self.mode = mode
        self.As, self.numes, self.rs, self.basers= self.load_dataset(self.mode, type, args)
        if type == 'all' and mode == 'train':
            Idx = random.sample(range(len(self.As)), k = len(self.As))
            self.As = [self.As[i] for i in Idx]
            self.numes = [self.numes[i] for i in Idx]
            self.rs = [self.rs[i] for i in Idx]
            self.basers = [self.basers[i] for i in Idx]
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)

    def load_dataset(self, mode, type, args):    



        As = pickle.load(open('./data/mech_{}/synthetic/As.pkl'.format(args.mech), 'rb'))
        numes = pickle.load(open('./data/mech_{}/synthetic/numes.pkl'.format(args.mech), 'rb'))
        rs = pickle.load(open('./data/mech_{}/synthetic/rs.pkl'.format(args.mech), 'rb'))
        basers = pickle.load(open('./data/mech_{}/synthetic/basers.pkl'.format(args.mech), 'rb'))

        return As, numes, rs, basers

    def __getitem__(self, index):

        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]

    def __len__(self):
        return len(self.As)
    
class NetDatasetEpidemic(Dataset):

    def __init__(self, mode, type, args):

        super(NetDatasetEpidemic, self).__init__()
        self.args = args
        self.mode = mode
        self.As, self.numes, self.rs, self.basers= self.load_dataset(self.mode, type, args)

        if len(self.As) > 1500:
            Idx = random.sample(range(len(self.As)), k = 1500)
            self.As = [self.As[i] for i in Idx]
            self.numes = [self.numes[i] for i in Idx]
            self.rs = [self.rs[i] for i in Idx]
            self.basers = [self.basers[i] for i in Idx]
        
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)
    
    def load_dataset(self, mode, type, args):
    
        As = pickle.load(open(f'./data/epidemic/{args.dataset_name}/total_As.pkl', 'rb'))
        numes = pickle.load(open(f'./data/epidemic/{args.dataset_name}/total_numericals_shrink.pkl', 'rb'))
        rs = pickle.load(open(f'./data/epidemic/{args.dataset_name}/total_rs.pkl', 'rb'))
        
        basers = [0] * len(As)
        return As, numes, rs, basers

    def __getitem__(self, index):
        return np.array(self.As[index]), np.array(self.numes[index][:, :self.args.hidden + 1, :]), self.rs[index], self.basers[index]
    
    def __len__(self):
        return len(self.As)


class TrajDatasetReal(Dataset):

    def __init__(self, mode, type, args):

        super(TrajDatasetReal, self).__init__()
        self.mode = mode
        self.numes, self.rs = self.load_dataset(self.mode, type, args)

        flattened_numes = [arr.flatten() for arr in self.numes]
        original_shapes = [arr.shape for arr in self.numes]
        conbined_array = np.concatenate(flattened_numes)

        self.min = conbined_array.min()
        self.max = conbined_array.max()

        cc = list(zip(self.numes, self.rs))
        random.shuffle(cc)
        self.numes[:], self.rs[:]= zip(*cc)

    def __len__(self):

        return len(self.rs)
    
    def __getitem__(self, index):

        nume = np.array(self.numes[index])

        nume = (nume - self.min) / (self.max - self.min)

        return nume, self.rs[index]
    
    def load_dataset(self, mode, type, args):

        if mode == 'all':
            
            numes = pickle.load(open(f'../data_micro/numes_new.pkl', 'rb'))
            rs = pickle.load(open(f'../data_micro/resilience_new.pkl', 'rb'))

        else:

            numes = pickle.load(open(f'../data_micro/numes_{mode}_new.pkl', 'rb'))
            rs = pickle.load(open(f'../data_micro/resilience_{mode}_new.pkl', 'rb'))

        return numes, rs
    

class TrajDatasetRealTopology(Dataset):

    def __init__(self, mode, type, args):

        super(TrajDatasetRealTopology, self).__init__()
        self.mode = mode
        self.numes, self.rs = self.load_dataset(self.mode, type, args)

        flattened_numes = [arr.flatten() for arr in self.numes]
        original_shapes = [arr.shape for arr in self.numes]
        conbined_array = np.concatenate(flattened_numes)

        self.min = conbined_array.min()
        self.max = conbined_array.max()

        cc = list(zip(self.numes, self.rs))
        random.shuffle(cc)
        self.numes[:], self.rs[:]= zip(*cc)

    def __len__(self):

        return len(self.rs)
    
    def __getitem__(self, index):

        nume = np.array(self.numes[index])

        nume = (nume - self.min) / (self.max - self.min)

        A = np.ones((nume.shape[-1], nume.shape[-1]))

        return A, nume, self.rs[index]
    
    def load_dataset(self, mode, type, args):

        if mode == 'all':
            
            numes = pickle.load(open(f'../data_micro/numes_new.pkl', 'rb'))
            rs = pickle.load(open(f'../data_micro/resilience_new.pkl', 'rb'))

        else:

            numes = pickle.load(open(f'../data_micro/numes_{mode}_new.pkl', 'rb'))
            rs = pickle.load(open(f'../data_micro/resilience_{mode}_new.pkl', 'rb'))

        return numes, rs

class TrajDataRealFromExist(Dataset):

    def __init__(self, numes, rs):

        self.numes = numes
        self.rs = rs
    
    def __len__(self):

        return len(self.rs)

    def __getitem__(self, idx):


        return np.array(self.numes[idx]), self.rs[idx]

class TrajDataRealFromExistTopology(Dataset):

    def __init__(self, numes, rs):

        super(TrajDataRealFromExistTopology, self).__init__()

        self.numes = numes
        self.rs = rs
    
    def __len__(self):

        return len(self.rs)

    def __getitem__(self, idx):

        A = np.ones((self.numes[idx].shape[-1], self.numes[idx].shape[-1]))

        return A, np.array(self.numes[idx]), self.rs[idx]
    


        

