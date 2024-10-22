import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import numpy as np
import random
from sklearn import metrics
import wandb
from tqdm import tqdm
from utils import *
import wandb
import pickle
from sklearn.metrics import precision_recall_curve

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

def process_instance(numericals, r_truth, args):

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    numericals = np.array(numericals)
    numericals = np.transpose(numericals, (0,2,1))
    add_nume = np.mean(numericals, axis=1, keepdims=True)
    numericals = np.concatenate((numericals, add_nume), axis=1)
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = torch.from_numpy(np.array(r_truth)).to(torch.float32).to(device)

    return numericals, r_truth



def process_instance_faketopo(A, numericals, r_truth, args):

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    A = np.array(A)
    add0 = np.ones((1, A.shape[0]))
    add1 = np.zeros((A.shape[0]+1, 1))
    A = np.concatenate((A, add0), axis=0)
    A = np.concatenate((A, add1), axis=1)

    A = normalized_laplacian(A)

    A = torch.from_numpy(A).to(torch.float32).to(device)

    numericals = np.array(numericals)
    numericals = np.transpose(numericals, (0,2,1))
    add_nume = np.mean(numericals, axis=1, keepdims=True)
    numericals = np.concatenate((numericals, add_nume), axis=1)
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = torch.from_numpy(np.array(r_truth)).to(torch.float32).to(device)

    return A, numericals, r_truth


def train_test_faketopo(model, train_subset, simu_valid_subset, real_valid_subset, train_loader, simu_valid_loader, test_loader, optimizer, criterion, args):

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.use_wandb:

        dir = wandb.run.dir
    else:
        dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_saver = CheckpointSaver(dirpath=os.path.join(dir, 'checkpoints'), decreasing=False, top_n=1)

    metric_monitor = MetricMonitor()

    for epoch in range(args.epoch):

        model.train()
        total_loss = 0
        for i, (A, numericals, r_truth) in tqdm(enumerate(train_loader), total=len(train_subset) // args.train_size + 1):
            
            A = A.numpy()
            A = A.squeeze(0)
            numericals = numericals.numpy()
            numericals = numericals.squeeze(0)
            A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)
            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))

            r_pred, _, __  = model(numericals_use[:,:,:1+args.train_seq_len], A)

            if not torch.isnan(r_pred):
                loss = criterion(r_pred, r_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        total_loss = total_loss / len(train_subset)
        print('Total Loss in Epoch {0}:'.format(epoch))
        print(total_loss)
        if args.use_wandb:
            wandb.log({'train_loss': total_loss, "epoch": epoch})

        r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

        with torch.no_grad():

            model.eval()

            test_loss = 0

            preds = []
            truths = []
            pred_labels = []

            for i, (A, numericals, r_truth) in tqdm(enumerate(simu_valid_loader), total=len(simu_valid_subset) // args.valid_size + 1):
                

                A = A.numpy()
                A = A.squeeze(0)
                numericals = numericals.numpy()
                numericals = numericals.squeeze(0)

                A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)
                numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))

                r_pred, _, __ = model(numericals_use[:,:,:1+args.test_seq_len], A)
                r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

                if not torch.isnan(r_pred):
                    
                    loss = criterion(r_pred, r_truth)
                    test_loss += loss.item()
                    preds.append(r_pred.item())
                    truths.append(r_truth.item())
                    pred_labels.append((r_pred.item() > args.threshold))

            test_loss = test_loss / len(simu_valid_subset)

            my_auc = metrics.roc_auc_score(truths, preds)
            my_f1 = metrics.f1_score(truths, pred_labels, average='weighted')
            my_acc = metrics.accuracy_score(truths, pred_labels)
            my_mcc = metrics.matthews_corrcoef(truths, pred_labels)

            
            # metric_monitor.update(my_f1, my_acc, my_mcc, my_auc, epoch)

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "Train Loss", "Simu valid Loss", "Accuracy", "AUC", "f1", "mcc", "Positive"]
            train_res.add_row([epoch, total_loss, test_loss, my_acc, my_auc, my_f1, my_mcc, sum(truths)/len(truths)])
            print(train_res)
            if args.use_wandb:
                wandb.log({'simu_valid_loss': test_loss, "epoch": epoch, "Simu_Acc": my_acc, "Simu_AUC": my_auc, "Simu_F1": my_f1, "Simu_Mcc": my_mcc, "Simu_positive": sum(truths)/len(truths)})
        
        with torch.no_grad():
            
            model.eval()

            test_loss = 0
            preds = []
            truths = []
            pred_labels = []

            for i, (A, numericals, r_truth) in tqdm(enumerate(test_loader), total=len(real_valid_subset) // args.valid_size + 1):
                

                A = A.numpy()
                A = A.squeeze(0)
                numericals = numericals.numpy()
                numericals = numericals.squeeze(0)

                A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)
                numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))

                r_pred, _, __ = model(numericals_use[:,:,:1+args.test_seq_len], A)
                r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

                if not torch.isnan(r_pred):
                    
                    loss = criterion(r_pred, r_truth)
                    test_loss += loss.item()
                    preds.append(r_pred.item())
                    truths.append(r_truth.item())
                    pred_labels.append((r_pred.item() > args.threshold))

            test_loss = test_loss / len(real_valid_subset)

            my_auc = metrics.roc_auc_score(truths, preds)
            my_f1 = metrics.f1_score(truths, pred_labels, average='weighted')
            my_acc = metrics.accuracy_score(truths, pred_labels)
            my_mcc = metrics.matthews_corrcoef(truths, pred_labels)
            checkpoint_saver(model, epoch, my_f1)
            # checkpoint_saver(model, epoch, my_f1)
            metric_monitor.update(my_f1, my_acc, my_mcc, my_auc, epoch)

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "Train Loss", "Real valid Loss", "Accuracy", "AUC", "f1", "mcc", "Positive"]
            train_res.add_row([epoch, total_loss, test_loss, my_acc, my_auc, my_f1, my_mcc, sum(truths)/len(truths)])
            print(train_res)
            if args.use_wandb:
                wandb.log({'Real_valid_loss': test_loss, "epoch": epoch, "Acc": my_acc, "AUC": my_auc, "F1": my_f1, "Mcc": my_mcc, "positive": sum(truths)/len(truths)})

    f1, acc, mcc, auc, epoch = metric_monitor.read()

    return [f1, acc, mcc, auc, epoch]


def test_faketopo_threshold(model, real_test_subset, test_loader, optimizer, criterion, args):

    device = args.device

    with torch.no_grad():
            
        model.eval()

        test_loss = 0
        preds = []
        truths = []
        pred_labels = []

        for i, (A, numericals, r_truth) in tqdm(enumerate(test_loader), total=len(real_test_subset) // args.test_size + 1):
            

            A = A.numpy()
            A = A.squeeze(0)
            numericals = numericals.numpy()
            numericals = numericals.squeeze(0)

            A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)
            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))

            r_pred, _, __ = model(numericals_use[:,:,:1+args.test_seq_len], A)
            r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

            if not torch.isnan(r_pred):
                
                loss = criterion(r_pred, r_truth)
                test_loss += loss.item()
                preds.append(r_pred.item())
                truths.append(r_truth.item())
                # pred_labels.append((r_pred.item() > args.threshold))

        precisions, recalls, thresholds = precision_recall_curve(truths, preds)
        f1_scores = 2 * precisions * recalls / (precisions + recalls)

        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        print('Best Threshold: ', best_threshold)
        y_pred_best = (np.array(preds) >= best_threshold).astype(int)

        # test_loss = test_loss / len(real_test_subset)

        my_auc = metrics.roc_auc_score(truths, y_pred_best)
        my_f1 = metrics.f1_score(truths, y_pred_best, average='weighted')
        my_acc = metrics.accuracy_score(truths, y_pred_best)
        my_mcc = metrics.matthews_corrcoef(truths, y_pred_best)

        # checkpoint_saver(model, epoch, my_f1)
        # metric_monitor.update(my_f1, my_acc, my_mcc, my_auc, epoch)

        train_res = PrettyTable()
        train_res.field_names = ["Test Loss", "Accuracy", "AUC", "f1", "mcc", "Positive"]
        train_res.add_row([test_loss, my_acc, my_auc, my_f1, my_mcc, sum(truths)/len(truths)])

        print(train_res)

        if args.use_wandb:

            pickle.dump(preds, open(os.path.join(wandb.run.dir, 'preds.pkl'), 'wb'))
            pickle.dump(truths, open(os.path.join(wandb.run.dir, 'truths.pkl'), 'wb'))
            
        return my_f1
        # wandb.log({'Real_valid_loss': test_loss, "epoch": epoch, "Acc": my_acc, "AUC": my_auc, "F1": my_f1, "Mcc": my_mcc, "positive": sum(truths)/len(truths)})
