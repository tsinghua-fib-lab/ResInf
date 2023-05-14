import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import random
from time import time
from collections import defaultdict
import warnings

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

class NetDataset(Dataset):
    def __init__(self, mode, type, args):
        super(NetDataset, self).__init__()
        self.As, self.numes, self.rs, self.basers= self.load_dataset(mode, type, args)
        if type == 'all':
            Idx = random.sample(range(len(self.As)), k = 1500)
            self.As = [self.As[i] for i in Idx]
            self.numes = [self.numes[i] for i in Idx]
            self.rs = [self.rs[i] for i in Idx]
            self.basers = [self.basers[i] for i in Idx]
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)
    def load_dataset(self, mode, type, args):
        As = json.load(open('../data/mech_{}_normal_As.json'.format(args.mech)))
        numes = json.load(open('../data/mech_{}_normal_numes.json'.format(args.mech)))
        rs = json.load(open('../data/mech_{}_normal_rs.json'.format(args.mech)))
        basers = json.load(open('../data/mech_{}_normal_basers.json'.format(args.mech)))
        return As, numes, rs, basers
    def __getitem__(self, index):
        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]
    def __len__(self):
        return len(self.As)
