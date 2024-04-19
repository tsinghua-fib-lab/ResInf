import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset
from sklearn.neighbors import KernelDensity
from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt


class NetDataset(Dataset):
    
    def __init__(self):
        super(NetDataset, self).__init__()
        self.As, self.numes, self.rs, self.basers= self.load_dataset()        
        Idx = random.sample(range(len(self.As)), k = 1500)
        self.As = [self.As[i] for i in Idx]
        self.numes = [self.numes[i] for i in Idx]
        self.rs = [self.rs[i] for i in Idx]
        self.basers = [self.basers[i] for i in Idx]
        cc = list(zip(self.As, self.numes, self.rs, self.basers))
        random.shuffle(cc)
        self.As[:], self.numes[:], self.rs[:], self.basers[:]= zip(*cc)
    
    def load_dataset(self):

        As = pickle.load(open('utils/As.pkl', 'rb'))
        numes = pickle.load(open('utils/numes.pkl', 'rb'))
        rs = pickle.load(open('utils/rs.pkl', 'rb'))
        basers = pickle.load(open('utils/basers.pkl', 'rb'))

        return As, numes, rs, basers

    def __getitem__(self, index):
        return np.array(self.As[index]), np.array(self.numes[index]), self.rs[index], self.basers[index]
    def __len__(self):
        return len(self.As)


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator
def process_instance(A, numericals, r_truth, r_base, args):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    A = np.array(A)
    add0 = np.ones((1, A.shape[0]))
    add1 = np.zeros((A.shape[0]+1, 1))
    A = np.concatenate((A, add0), axis=0)
    A = np.concatenate((A, add1), axis=1)
    assert A.shape[0] == A.shape[1]
    A = normalized_laplacian(A)
    A = torch.from_numpy(A).to(torch.float32).to(device)
    numericals = np.array(numericals)
    numericals = np.transpose(numericals, (0,2,1))
    add_nume = np.mean(numericals, axis=1, keepdims=True)
    numericals = np.concatenate((numericals, add_nume), axis=1)
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = torch.from_numpy(np.array(r_truth)).to(torch.float32).to(device)
    r_base = torch.from_numpy(np.array(r_base)).to(torch.float32).to(device)
    return A, numericals, r_truth, r_base

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                    x0, y0, width, height, fontsize, trans):
        if len(orig_handle) == 2:
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                            linestyle=orig_handle[1], color=orig_handle[0])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                            color=orig_handle[0])
            return [l1, l2]
        else:
            l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height],
                            linestyle='--', color=orig_handle[0])
            return [l1]