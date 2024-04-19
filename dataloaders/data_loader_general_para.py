import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch.utils.data import Dataset
import pickle
import json

class NetTrainDataset(Dataset):
    def __init__(self, args, train_group):
        super(NetTrainDataset, self).__init__()

        self.train_group = train_group

        self.As, self.numes, self.rs, self.basers= self.load_dataset(args.mech)

        Idx = random.sample(range(len(self.As)), k = 2000)
        self.As = [self.As[i] for i in Idx]
        self.numes = [self.numes[i] for i in Idx]
        self.rs = [self.rs[i] for i in Idx]
        self.basers = [self.basers[i] for i in Idx]

        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)
    def load_dataset(self, mech):

        As = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[0]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[1]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[2]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[3]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[4]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[5]}/As.pkl', 'rb'))
        numes = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[0]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[1]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[2]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[3]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[4]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[5]}/numes_shrink.pkl', 'rb'))
        rs = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[0]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[1]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[2]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[3]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[4]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[5]}/rs.pkl', 'rb'))
        basers = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[0]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[1]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[2]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[3]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[4]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.train_group[5]}/basers.pkl', 'rb'))
        
        return As, numes, rs, basers
   
    def __getitem__(self, index):
        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]
    
    def __len__(self):
        return len(self.As)

class NetTestDataset(Dataset):
    def __init__(self, mode, args, test_group):
        super(NetTestDataset, self).__init__()
        self.args = args
        self.test_group = test_group
        self.As, self.numes, self.rs, self.basers= self.load_dataset(args.mech)
        
        Idx = random.sample(range(len(self.As)), k = int(mode * 0.2)+1)
        self.As = [self.As[i] for i in Idx]
        self.numes = [self.numes[i] for i in Idx]
        self.rs = [self.rs[i] for i in Idx]
        self.basers = [self.basers[i] for i in Idx]
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)
    
    def __getitem__(self, index):
        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]
    def __len__(self):
        return len(self.As)

    def load_dataset(self,mech):
    
        As = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[0]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[1]}/As.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[2]}/As.pkl', 'rb'))
        numes = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[0]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[1]}/numes_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[2]}/numes_shrink.pkl', 'rb'))
        rs = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[0]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[1]}/rs.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[2]}/rs.pkl', 'rb'))
        basers = pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[0]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[1]}/basers.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech}/general_para/idx_{self.test_group[2]}/basers.pkl', 'rb'))

        return As, numes, rs, basers