import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataloaders.data_loader_hyper import NetDataset, NetDataset2
from torch.utils.data import DataLoader
from model.resinf import ResInf
from engine import *
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=6,
                    help='Number of hidden units.')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2021, help='Random Seed')
parser.add_argument('--T', type=float, default=200., help='Terminal Time')
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_adj')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--train_size', type=int, default=1)
parser.add_argument('--valid_size', type=int, default=1)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--rand_guess', type=bool, default=False)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--use', type=str, default='start')
parser.add_argument('--type', type=str, default='all')
parser.add_argument('--K', type=int, default=11)
parser.add_argument('--comment', type=str, default='normal')
parser.add_argument('--mech', type=int, default=1)
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
parser.add_argument('--finetune',type=int, default=0)
parser.add_argument('--valid_every', type=int, default=5)
parser.add_argument('--save_every', action='store_true')
parser.add_argument('--use_wandb', action='store_true')

args = parser.parse_args()
if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
if __name__ == '__main__':

    if args.use_wandb:
        wandb.init(project="resinf", name=str(args.mech) + 'ref')
    epsilon = 1e-6
    seed = 2021
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = NetDataset2(mode = 'train', type=args.type, args=args)

    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.valid_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.test_size)

    input_size = 1

    if args.use_model == 'resinf':
        if args.mech == 1:
            args.K = 10
            args.layers = 3
            args.emb_size = 8
            args.hidden_layers_num = 1
            args.trans_layers = 1
            args.trans_emb_size = 32
        elif args.mech == 2:
            args.K = 5
            args.layers = 3
            args.emb_size = 16
            args.hidden_layers_num = 1
            args.trans_layers = 1
            args.trans_emb_size = 8
        elif args.mech == 3 or args.mech == 0:
            args.K = 11
            args.layers = 3
            args.emb_size = 8
            args.hidden_layers_num = 1
            args.trans_layers = 1
            args.trans_emb_size = 8
            
    model = ResInf(input_plane=args.K, seq_len = args.hidden, trans_layers=args.trans_layers, gcn_layers=args.layers, hidden_layers=args.hidden_layers_num, gcn_emb_size=args.emb_size, trans_emb_size=args.trans_emb_size, pool_type=args.pool_type, args=args,n_heads=args.n_heads).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    final_epoch, final_total_loss = train_valid(model, train_dataset, valid_dataset, train_data_loader, valid_data_loader, optimizer, criterion, args)
    test(model, test_data_loader, criterion, args, final_epoch, final_total_loss)