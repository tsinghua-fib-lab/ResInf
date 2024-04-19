import numpy as np
import random
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch.utils.data import Dataset



class NetDataset_new(Dataset):
    def __init__(self, mode, mech, args, min=None, max=None, ori_test=False):
        super(NetDataset_new, self).__init__()

        self.mode = mode

        self.ori_test = ori_test

        self.args = args

        self.mech_dict = {1: 'mutualistic', 2:'regulatory', 3:'neuronal', 4: 'sis', 5: 'inhibitory'}

        self.As, self.numes, self.rs, self.basers = self.load_dataset(mech)

        if self.mode == 'train':

            self.k = 1600

        elif self.mode == 'val':
            
            self.k = 400

        
        else:
            raise NotImplementedError
        
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)

        Idx = random.sample(range(len(self.As)), k = self.k)
        self.As = [self.As[i] for i in Idx]
        self.numes = [self.numes[i] for i in Idx]

        
        flattened_numes = [arr.flatten() for arr in self.numes]
        original_shapes = [arr.shape for arr in self.numes]
        conbined_array = np.concatenate(flattened_numes)
        if self.mode == 'train':
            
            min_val = conbined_array.min()
            self.min = min_val
            max_val = conbined_array.max()
            self.max = max_val

        else:
            self.min = min
            self.max = max

        normalized_array = (conbined_array - self.min) / (self.max - self.min)

        normalized_arrays = []
        start = 0
        for shape in original_shapes:
            size = np.prod(shape)
            normalized_arrays.append(normalized_array[start:start + size].reshape(shape))
            start += size

        self.numes = normalized_arrays

        self.rs = [self.rs[i] for i in Idx]
        self.basers = [self.basers[i] for i in Idx]

        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)


    def load_dataset(self, mech):

        if type(mech) == int:

            mech = [mech]
            
        As, numes, rs, basers = [], [], [], []

        for i in range(len(mech)):

            if self.mode == 'val':

                As = As + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_2.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_3.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_4.pkl', 'rb'))
                numes = numes + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_2_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_3_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_4_shrink.pkl', 'rb'))
                rs = rs + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_2.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_3.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_4.pkl', 'rb'))
                basers = [0] * len(As)

                print(f'Validation data load mech:{mech[i]}.')
            
            elif self.mode == 'test':

                As = As + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_2.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_3.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_4.pkl', 'rb'))
                numes = numes + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_2_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_3_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_4_shrink.pkl', 'rb'))
                rs = rs + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_2.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_3.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_4.pkl', 'rb'))
                basers = [0] * len(As)


                print(f'Test data load mech:{mech[i]}.')

            else:

                if mech[i] == 4:

                    As = As + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_d_5.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_d_6.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_d_7.pkl', 'rb'))
                    numes = numes + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_5_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_6_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_7_shrink.pkl', 'rb'))
                    rs = rs + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_d_5.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_d_6.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_d_7.pkl', 'rb'))
                    basers = [0] * len(As)
                
                elif mech[i] == 5:

                    As = As + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_b_0.1_k_8_c_4.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_b_0.1_k_8_c_5.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/As_b_0.01_k_8_c_5.pkl', 'rb'))
                    numes = numes + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_b_0.1_k_8_c_4_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_b_0.1_k_8_c_5_shrink.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/numes_b_0.01_k_8_c_5_shrink.pkl', 'rb'))
                    rs = rs + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_b_0.1_k_8_c_4.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_b_0.1_k_8_c_5.pkl', 'rb')) + pickle.load(open(f'./data/mech_{mech[i]}/general_equ/rs_b_0.01_k_8_c_5.pkl', 'rb'))
                    basers = [0] * len(As)

                else:
                    raise NotImplementedError
                    

                print(f'Train data load mech:{mech[i]}.')

        return As, numes, rs, basers
            
    def __getitem__(self, index):

        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]
    
    def __len__(self):
        
        return len(self.As)