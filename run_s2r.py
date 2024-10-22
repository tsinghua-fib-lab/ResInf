
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataloaders.data_loader_s2r import S2R_Dataset_train, S2R_Dataset_test
from torch.utils.data import DataLoader
from model.resinf_foreal import ResInf
from engine_s2r import *
import wandb
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=6,
                    help='Number of hidden units.')
parser.add_argument('--time_tick', type=int, default=100) # default=10)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=2021, help='Random Seed')
parser.add_argument('--T', type=float, default=200., help='Terminal Time')
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_adj')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--train_size', type=int, default=1)
parser.add_argument('--valid_size', type=int, default=1)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--rand_guess', type=bool, default=False)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--use', type=str, default='start')
parser.add_argument('--type', type=str, default='node')
parser.add_argument('--causal', type=int, default=0)
parser.add_argument('--K', type=int, default=11)
parser.add_argument('--comment', type=str, default='normal')
parser.add_argument('--train_mech', type=str, default='[4]')
parser.add_argument('--val_mech', type=str, default='1')
parser.add_argument('--asso', type=int, default=0)
parser.add_argument('--use_model',type=str, default='resinf')
parser.add_argument('--decompo', type=str, default='None')
parser.add_argument('--cross', type=int, default=0)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--emb_size',type=int,default=8)
parser.add_argument('--hidden_layers_num', type=int, default=1)
parser.add_argument('--pool_type', type=str, default='virtual')
parser.add_argument('--pool_arch', type=str, default='global')
parser.add_argument('--trans_layers', type=int, default=1)
parser.add_argument('--trans_emb_size',type=int, default=8)
parser.add_argument('--n_heads',type=int, default=4)
parser.add_argument('--name', type=str, default='exp')
parser.add_argument('--ori_test', action='store_true')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--save_every', action='store_true')
parser.add_argument('--train_seq_len', type=int, default=2)
parser.add_argument('--test_seq_len', type=int, default=2)
parser.add_argument('--dim_feedforward', type=int, default=8)
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--ckpt_load_path', type=str, default='')
parser.add_argument('--use_wandb', action='store_true')
args = parser.parse_args()

if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

def set_cpu_num(cpu_num):

    if cpu_num <= 0: 
        return

    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.set_num_interop_threads(1)

if __name__ == '__main__':


    set_cpu_num(1)
    
    epsilon = 1e-6
    seed = 2022
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



    if args.use_wandb:

        wandb.init(project="s2r-validation")



    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    args.device = device

    input_size = 1
    args.K = 5
    args.hidden = 2
    args.layers = 2
    args.trans_layers = 3
    args.hidden_layers_num = 3
    args.emb_size = 16
    args.trans_emb_size = 16
    args.n_heads = 4

    if not args.test_only:
    
        simu_dataset = S2R_Dataset_train(mode = 'simulate', args=args)
        simu_valid_length = int(len(simu_dataset) * 0.1)
        train_dataset, simu_valid_dataset = torch.utils.data.random_split(simu_dataset, [len(simu_dataset) - simu_valid_length, simu_valid_length])
        train_loader = DataLoader(train_dataset, batch_size=args.train_size, shuffle=True)
        simu_valid_loader = DataLoader(simu_valid_dataset, batch_size=args.valid_size, shuffle=True)
    real_dataset = S2R_Dataset_test(mode = 'real_sample', args=args)

    
    real_dataset_loader = DataLoader(real_dataset, batch_size=args.valid_size, shuffle=False)


    model = ResInf(input_plane=args.K, seq_len = args.train_seq_len + 1, trans_layers=args.trans_layers, gcn_layers=args.layers, hidden_layers=args.hidden_layers_num, gcn_emb_size=args.emb_size, trans_emb_size=args.trans_emb_size, pool_type=args.pool_type, args=args,n_heads=args.n_heads).to(device)

    
    for p in model.parameters():

        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    if not args.test_only:

        final_result = train_test_faketopo(model, train_dataset, simu_valid_dataset, real_dataset, train_loader, simu_valid_loader, real_dataset_loader, optimizer, criterion, args)
    
    else:

        model.load_state_dict(torch.load(args.ckpt_load_path, map_location=device))

        final_result = test_faketopo_threshold(model, real_dataset, real_dataset_loader, optimizer, criterion, args)
    


