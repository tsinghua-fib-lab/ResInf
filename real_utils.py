from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from engine_real import *
from model.resinf_foreal import ResInf
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
from sklearn.model_selection import train_test_split
from dataloaders.data_loader_hyper import TrajDataRealFromExistTopology


def stratified_k_fold_cross_validation(args, k, device):
    
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    results = []

        
    numes = pickle.load(open('./data/micro/numes.pkl', 'rb'))
    rs = pickle.load(open('./data/micro/rs.pkl', 'rb'))

            
    new_numes = []

    for nume in numes:

        nume = np.array(nume)

        new_numes.append(nume)
    
    cc = list(zip(new_numes, rs))
    random.shuffle(cc)
    new_numes[:], rs[:] = zip(*cc)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(new_numes, rs)):

        run_name = f"SK-fold_{fold + 1}"
        if args.use_wandb:
            wandb.init(project=f'real-stratified-k-fold-num-{args.fold_num}-{args.use_model}-1', name=run_name, reinit=True)

        train_numes = [new_numes[i] for i in train_ids]
        train_rs = [rs[i] for i in train_ids]

        test_numes = [new_numes[i] for i in test_ids]
        test_rs = [rs[i] for i in test_ids]


        train_subset = TrajDataRealFromExistTopology(train_numes, train_rs)
        test_subset = TrajDataRealFromExistTopology(test_numes, test_rs)

        train_loader = DataLoader(train_subset, batch_size=args.train_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.test_size, shuffle=False)

        args.K = 2
        args.layers = 1
        args.trans_layers = 1
        args.hidden_layers_num = 2
        args.emb_size = 4
        args.trans_emb_size = 8
        args.n_heads = 1

        model = ResInf(input_plane=args.K, seq_len = args.hidden, trans_layers=args.trans_layers, gcn_layers=args.layers, hidden_layers=args.hidden_layers_num, gcn_emb_size=args.emb_size, trans_emb_size=args.trans_emb_size, pool_type=args.pool_type, args=args,n_heads=args.n_heads).to(device)
        
        print(model)
        print(args)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        final_result = train_test_faketopo(model, train_subset, test_subset, train_loader, test_loader, optimizer, criterion, args)

        results.append(final_result)

        wandb.finish()

    return results



    






    

