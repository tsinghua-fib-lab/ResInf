import argparse
import numpy as np
import torch
import random
from real_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--time_tick', type=int, default=100) # default=10)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=2021, help='Random Seed')
parser.add_argument('--T', type=float, default=200., help='Terminal Time')
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_adj')
parser.add_argument('--epoch', type=int, default=50)
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
parser.add_argument('--mech', type=int, default=1)
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
parser.add_argument('--finetune',type=int, default=0)
parser.add_argument('--train_type', type=str, default='kfold')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--fold_num', type=int, default=8)
parser.add_argument('--use_logitloss', action='store_true')
parser.add_argument('--use_wandb', action='store_true')
args = parser.parse_args()

if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if __name__ == '__main__':

    torch.set_num_threads(1)

    all_f1_score = []

    for _ in range(10):
        
        seed = np.random.randint(0, 1000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        results = stratified_k_fold_cross_validation(args, args.fold_num, device)
        if args.use_wandb:
            wandb.init(project=f'real-stratified-k-fold-num-{args.fold_num}', name='statistics', reinit=True)
        results = np.array(results)
        f1s = results[:, 0]
        accs = results[:, 1]
        mccs = results[:, 2]
        aucs = results[:, 3]
        print(f'F1: {f1s.mean()}±{f1s.std()}')
        all_f1_score.append(f1s.mean())
        print(f'Acc: {accs.mean()}±{accs.std()}')
        print(f'MCC: {mccs.mean()}±{mccs.std()}')
        print(f'AUC: {aucs.mean()}±{aucs.std()}')
        wandb.log({'F1': f1s.mean(), 'F1_std': f1s.std()})
        wandb.log({'Acc': accs.mean(), 'Acc_std': accs.std()})
        wandb.log({'MCC': mccs.mean(), 'MCC_std': mccs.std()})
        wandb.log({'AUC': aucs.mean(), 'AUC_std': aucs.std()})
        
    print('All f1 mean:', np.array(all_f1_score).mean())
    print('All f1 std:', np.array(all_f1_score).std())
    


    
