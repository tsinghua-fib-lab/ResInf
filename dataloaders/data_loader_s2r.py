import numpy as np
import random
import pickle
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from torch.utils.data import Dataset


class S2R_Dataset_train(Dataset):

    def __init__(self, mode, args):

        super().__init__()
        self.args = args

        self.mode = mode

        self.As, self.numes, self.rs, self.basers = self.load_dataset()

    
        cc = list(zip(self.As, self.numes, self.rs, self.basers))

        random.shuffle(cc)

        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)

        flattened_train = [arr.flatten() for arr in self.numes]
        original_shapes = [arr.shape for arr in self.numes]
        conbined_array = np.concatenate(flattened_train)

        # conbined_array = np.log1p(conbined_array) ## logrithm transformation

        min_val = conbined_array.min()
        self.min = min_val
        max_val = conbined_array.max()
        self.max = max_val
        mean_val = conbined_array.mean()
        self.mean = mean_val
        std_val = conbined_array.std()
        self.std = std_val
        normalized_array = (conbined_array - self.mean) / self.std
        normalized_arrays = []
        start = 0
        for shape in original_shapes:
            size = np.prod(shape)
            normalized_arrays.append(normalized_array[start:start + size].reshape(shape))
            start += size
        self.numes = normalized_arrays

    def load_dataset(self):

        
        As_path = f'./data/data_synthesis_s2r/As_sis_synthesis.pkl'
        numes_path = f'./data/data_synthesis_s2r/numericals_sis_synthesis.pkl'
        rs_path = f'./data/data_synthesis_s2r/rs_sis_synthesis.pkl'
        
        As = pickle.load(open(As_path, 'rb'))
        numes = pickle.load(open(numes_path, 'rb'))
        rs = pickle.load(open(rs_path, 'rb'))
        basers = [0] * len(As)

        return As, numes, rs, basers


    def __len__(self):


        return len(self.rs)

    def __getitem__(self, index):
        
        nume = np.array(self.numes[index])

    
        return np.array(self.As[index]), nume, self.rs[index]
        

class S2R_Dataset_test(Dataset):

    def __init__(self, mode, args, min=None, max=None, mean=None, std=None):

        super().__init__()

        self.args = args

        self.numes, self.rs, self.basers = self.load_dataset()


        flattened_test = [arr.flatten() for arr in self.numes]
        original_shapes_test = [arr.shape for arr in self.numes]
        conbined_array_test = np.concatenate(flattened_test)


        min_test = conbined_array_test.min()
        self.min = min_test
        max_test = conbined_array_test.max()
        self.max = max_test
        mean_test = conbined_array_test.mean()
        self.mean = mean_test
        std_test = conbined_array_test.std()
        self.std = std_test

        normalized_array_test = (conbined_array_test - self.mean) / self.std
        normalized_arrays_test = []
        start = 0
        for shape in original_shapes_test:
            size = np.prod(shape)
            normalized_arrays_test.append(normalized_array_test[start:start + size].reshape(shape))
            start += size
        self.numes = normalized_arrays_test


        if not args.test_only:
        
            cc = list(zip(self.numes, self.rs, self.basers))

            random.shuffle(cc)

            self.numes[:], self.rs[:], self.basers[:]= zip(*cc)

    def load_dataset(self):

        As = []
        numes = []
        rs = []
        basers = []

        numes = pickle.load(open('./data/data_micro_s2r/numes_micro.pkl', 'rb'))
        rs = pickle.load(open('./data/data_micro_s2r/rs_micro.pkl', 'rb'))
        basers = [0] * len(rs)

        return numes, rs, basers
        
        
    def __len__(self):

        return len(self.rs)

    def __getitem__(self, index):

        nume = np.array(self.numes[index])

        A = np.ones((nume.shape[-1], nume.shape[-1]))

        return A, nume, self.rs[index]



