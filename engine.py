import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import numpy as np
import random
from sklearn import metrics
import wandb
from tqdm import tqdm
from utils import *
import os

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

def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def process_instance(A, numericals, r_truth, r_base, args):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    A = np.array(A)
    if args.use_model != 'new' or (args.use_model == 'new' and args.pool_type == 'virtual'):
        add0 = np.ones((1, A.shape[0]))
        add1 = np.zeros((A.shape[0]+1, 1))
        A = np.concatenate((A, add0), axis=0)
        A = np.concatenate((A, add1), axis=1)
    assert A.shape[0] == A.shape[1]
    if args.operator == 'norm_lap':
        A = normalized_laplacian(A)
    elif args.operator == 'lap':
        D = np.diag(A.sum(axis=1))
        A = D - A
    elif args.operator == 'kipf':
        A = zipf_smoothing(A)
    elif args.operator == 'norm_adj':
        A = normalized_adj(A)
    else:
        raise NotImplementedError

    A = torch.from_numpy(A).to(torch.float32).to(device)
    numericals = np.array(numericals)
    numericals = np.transpose(numericals, (0,2,1))
    if args.use_model != 'new' or (args.use_model == 'new' and args.pool_type == 'virtual'):
        add_nume = np.mean(numericals, axis=1, keepdims=True)
        numericals = np.concatenate((numericals, add_nume), axis=1)
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = torch.from_numpy(np.array(r_truth)).to(torch.float32).to(device)
    r_base = torch.from_numpy(np.array(r_base)).to(torch.float32).to(device)
    return A, numericals, r_truth, r_base


def train_valid(model, train_dataset, valid_dataset, train_dataloader, valid_dataloader, optimizer, criterion, args, epsilon=1e-6):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.use_wandb:

        dir = wandb.run.dir
    else:
        dir = os.path.dirname(os.path.abspath(__file__))

    make_model_dirs(dir)

    if args.save_every:
        print('Save models Every Epoch.')
    checkpoint_saver = CheckpointSaver(dirpath=os.path.join(dir, 'checkpoints'), decreasing=False, top_n=1, save_every=args.save_every)
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for i, (A, numericals, r_truth, r_base) in tqdm(enumerate(train_dataloader), total=len(train_dataset)//args.train_size+1):
            A = A.numpy()
            numericals = numericals.numpy()
            A = A.squeeze(0)
            numericals = numericals.squeeze(0)
            check = np.any(np.array(A))
            if not check:
                continue

            A, numericals, r_truth, r_base = process_instance(A, numericals, r_truth, r_base, args)
            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))
            if args.rand_guess:
                r_pred = torch.randint(low=0, high=2,size=(1,)).to(torch.float32).to(device)
            else:
                    # print(numericals_use.shape)
                    # print(A.shape)                    
                r_pred, _, _= model(numericals_use[:,:,1:1+args.hidden], A)
            r_pred = min(r_pred + epsilon, torch.FloatTensor([1]).to(device))
            if not torch.isnan(r_pred):
                loss = criterion(r_pred, r_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        total_loss = total_loss/len(train_dataset)
        print('Total Loss in Epoch {0}:'.format(epoch))
        print(total_loss)

        if args.use_wandb:
            wandb.log({'train_loss': total_loss, "epoch": epoch})
        
        if epoch > 0 and epoch % args.valid_every == 0:
            with torch.no_grad():
                all_count = 0
                true_count = 0
                valid_loss = 0
                true_base = 0
                positive_count = 0
                preds = []
                truths = []
                base_preds = []
                pred_labels = []
                high_dim_vecs = torch.zeros(1, 1).to(device)
                for i, (A, numericals, r_truth, r_base) in tqdm(enumerate(valid_dataloader), total=len(valid_dataset)//args.valid_size+1):
                    A = A.numpy()
                    numericals = numericals.numpy()
                    A = A.squeeze(0)
                    numericals = numericals.squeeze(0)
                    check = np.any(np.array(A))
                    if not check:
                        continue
                    all_count += 1
                    A, numericals, r_truth, r_base = process_instance(A, numericals, r_truth, r_base, args)
                    numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))
                    if args.rand_guess:
                        r_pred = torch.randint(low=0, high=2,size=(1,)).to(torch.float32).to(device)
                    else:
                        r_pred,high_dim_vec,_ = model(numericals_use[:,:,1:1+args.hidden], A)
                        high_dim_vec = high_dim_vec.contiguous().view(1, -1)

                    r_pred = min(r_pred + epsilon, torch.FloatTensor([1]).to(device))
                    if not torch.isnan(r_pred):
                        loss = criterion(r_pred, r_truth)
                        valid_loss += loss.item()
                        preds.append(r_pred.item())
                        truths.append(r_truth.item())
                        base_preds.append(r_base.item())
                        pred_labels.append(round(r_pred.item()))
                        if round(r_pred.item()) == r_truth.item():
                            true_count += 1
                        if (r_base.item()) == r_truth.item():
                            true_base += 1
                        if r_truth.item() == 1:
                            positive_count += 1
                            
                        else:
                            pass
                valid_loss = valid_loss/len(valid_dataset)
                acc = true_count/all_count
                acc_base = true_base/all_count
                positive = positive_count/all_count
                preds = np.array(preds)
                truths = np.array(truths)
                base_preds = np.array(base_preds)
                my_f1 = metrics.f1_score(truths, preds.round(), average='weighted')
                base_f1 = metrics.f1_score(truths,base_preds, average='weighted')
                my_mcc = metrics.matthews_corrcoef(truths, preds.round())
                base_mcc = metrics.matthews_corrcoef(truths, base_preds)
                high_dim_vecs = high_dim_vecs.detach().cpu().numpy()
                high_dim_vecs = np.array(high_dim_vecs)
                checkpoint_saver(model, epoch, my_f1)
                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "Train Loss", "Valid Loss", "Accuracy", "Accuracy_bl", "f1", "f1_bl", "mcc", "mcc_bl", "Positive"]
                train_res.add_row([epoch, total_loss, valid_loss, acc, acc_base, my_f1, base_f1, my_mcc, base_mcc, positive])
                print(train_res)

                if args.use_wandb:
                    wandb.log({'val_loss': valid_loss, 'val_acc': acc, 'val_acc_bl': acc_base, 'val_f1': my_f1, 'val_f1_bl': base_f1, 'val_mcc': my_mcc, 'val_mcc_bl': base_mcc, 'val_positive': positive, 'epoch': epoch})
    return epoch, total_loss

def test(model, test_dataloader, criterion, args, epoch, total_loss, epsilon=1e-6):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        all_count = 0
        true_count = 0
        true_base = 0
        test_loss = 0
        positive_count = 0
        preds = []
        truths = []
        base_preds = []

        pred_labels = []
        
        high_dim_vecs = torch.zeros(1,1).to(device)

        for i, (A, numericals, r_truth, r_base) in tqdm(enumerate(test_dataloader)):
            A = A.numpy()
            numericals = numericals.numpy()
            A = A.squeeze(0)
            numericals = numericals.squeeze(0)
            check = np.any(np.array(A))
            if not check:
                continue
            all_count += 1
            A, numericals, r_truth, r_base= process_instance(A, numericals, r_truth, r_base, args)
            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))
            if args.rand_guess:
                r_pred = torch.randint(low=0, high=2,size=(1,)).to(torch.float32).to(device)
            else:

                
                r_pred, high_dim_vec,_ = model(numericals_use[:,:,1:1+args.hidden], A)
               
                high_dim_vec = high_dim_vec.contiguous().view(1, -1)


            r_pred = min(r_pred + epsilon, torch.FloatTensor([1]).to(device))
            if not torch.isnan(r_pred):
                loss = criterion(r_pred, r_truth)
                test_loss += loss.item()
                preds.append(r_pred.item())
                truths.append(r_truth.item())
                base_preds.append(r_base.item())
                pred_labels.append(round(r_pred.item()))
                if round(r_pred.item()) == r_truth.item():
                    true_count += 1
                if r_truth.item() == r_base.item():
                    true_base += 1
                if r_truth.item() == 1:
                    positive_count += 1
                    
                else:
                    pass
        test_loss = test_loss/len(truths)
        acc = true_count/all_count
        acc_base = true_base/all_count
        positive = positive_count/all_count
        preds = np.array(preds)
        truths = np.array(truths)
        base_preds = np.array(base_preds)
        my_f1 = metrics.f1_score(truths, preds.round(), average='weighted')
        base_f1 = metrics.f1_score(truths,base_preds, average='weighted')
        my_mcc = metrics.matthews_corrcoef(truths, preds.round())
        base_mcc = metrics.matthews_corrcoef(truths, base_preds)
        high_dim_vecs = high_dim_vecs.detach().cpu().numpy()
        high_dim_vecs = np.array(high_dim_vecs)
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "Train Loss", "Test Loss", "Accuracy", "Accuracy_bl", "f1", "f1_bl", "mcc", "mcc_bl", "Positive"]
        train_res.add_row([epoch, total_loss, test_loss, acc, acc_base, my_f1, base_f1, my_mcc, base_mcc, positive])
        print(train_res)


def test_forplot(model, test_dataloader, criterion, args, epoch, total_loss, epsilon=1e-6):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        all_count = 0
        true_count = 0
        true_base = 0
        test_loss = 0
        positive_count = 0
        preds = []
        truths = []
        base_preds = []
        cos_neg = []
        base_cos_neg = []
        cos_pos = []
        base_cos_pos = []
        dis_neg = []
        base_dis_neg = []
        dis_pos = []
        base_dis_pos = []
        pred_labels = []
        if args.use_model == 'single' or args.use_model == 'sig_classi' or args.use_model == 'sig_mean' or args.use_model == 'sig_multiclassi' or args.use_model == 'sig_woat' or args.use_model == 'sig_wosd' or args.use_model == 'new' or args.use_model == 'sig_max' or args.use_model == 'transgnn' or args.use_model == 'IND' or args.use_model == 'FC':
            high_dim_vecs = torch.zeros(1,1).to(device)
        else:
            high_dim_vecs = torch.zeros(1,2).to(device)
        for i, (A, numericals, r_truth, r_base) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            A = A.numpy()
            numericals = numericals.numpy()
            A = A.squeeze(0)
            numericals = numericals.squeeze(0)
            check = np.any(np.array(A))
            if not check:
                continue
            all_count += 1
            A, numericals, r_truth, r_base= process_instance(A, numericals, r_truth, r_base, args)

            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))
            if args.rand_guess:
                r_pred = torch.randint(low=0, high=2,size=(1,)).to(torch.float32).to(device)
            else:
               
                r_pred, high_dim_vec,_ = model(numericals_use[:,:,1:1+args.hidden], A)
                
                high_dim_vec = high_dim_vec.contiguous().view(1, -1)
                
            r_pred = min(r_pred + epsilon, torch.FloatTensor([1]).to(device))
            if not torch.isnan(r_pred):
                loss = criterion(r_pred, r_truth)
                test_loss += loss.item()
                preds.append(r_pred.item())
                truths.append(r_truth.item())
                base_preds.append(r_base.item())
                pred_labels.append(round(r_pred.item()))
                if round(r_pred.item()) == r_truth.item():
                    true_count += 1
                if r_truth.item() == r_base.item():
                    true_base += 1
                if r_truth.item() == 1:
                    positive_count += 1   
                else:
                    pass
        test_loss = test_loss/len(truths)
        acc = true_count/all_count
        acc_base = true_base/all_count
        positive = positive_count/all_count
        preds = np.array(preds)
        truths = np.array(truths)
        base_preds = np.array(base_preds)
        my_f1 = metrics.f1_score(truths, preds.round(), average='weighted')
        base_f1 = metrics.f1_score(truths,base_preds, average='weighted')
        my_mcc = metrics.matthews_corrcoef(truths, preds.round())
        base_mcc = metrics.matthews_corrcoef(truths, base_preds)
        high_dim_vecs = high_dim_vecs.detach().cpu().numpy()
        high_dim_vecs = np.array(high_dim_vecs)
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "Train Loss", "Test Loss", "Accuracy", "Accuracy_bl", "f1", "f1_bl", "mcc", "mcc_bl", "Positive"]
        train_res.add_row([epoch, total_loss, test_loss, acc, acc_base, my_f1, base_f1, my_mcc, base_mcc, positive])
        print(train_res)

    return preds, truths, preds.round(), my_f1, acc, my_mcc