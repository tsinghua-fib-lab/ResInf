import os
import numpy as np
import logging
import wandb
import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj, to_dense_batch



class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5, save_every=False, not_save=False):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): 
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        self.save_every = save_every
        self.not_save = not_save
        
    def __call__(self, model, epoch, metric_val, final_epoch=False):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')

        if self.not_save:
            save = False
        else:
            if self.save_every:
                save = True
            
            elif final_epoch:
                save = True
                
            else:
                save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            print(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            # self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        print(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

class MetricMonitor:

    def __init__(self):

        self.f1 = -np.inf
        self.acc = -np.inf
        self.mcc = -np.inf
        self.auc = -np.inf
        self.epoch = 0
    
    def update(self, f1, acc, mcc, auc, epoch):

        if f1 > self.f1:
            self.f1 = f1
            self.acc = acc
            self.mcc = mcc
            self.auc = auc
            self.epoch = epoch
            
    def read(self):

        return self.f1, self.acc, self.mcc, self.auc, self.epoch

class PlaceHolder:

    def __init__(self, X, E, y):

        self.X = X
        self.E = E
        self.y = y
    
    def type_as(self, x: torch.Tensor):

        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)

        return self

    def mask(self, node_mask, collapse=False):

        x_mask = node_mask.unsqueeze(-1).unsqueeze(-1) # (bs, n, 1, 1)
        e_mask1 = node_mask.unsqueeze(-1).unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = node_mask.unsqueeze(-1).unsqueeze(1)             # bs, 1, n, 1

        self.X = self.X * x_mask
        self.E = self.E * (e_mask1 * e_mask2).squeeze(-1) # bs, n, n

        return self


def make_model_dirs(path):

    model_path = os.path.join(path, os.path.pardir, 'checkpoints')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

# def to_dense(x, )
        
def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

def to_dense_dt(x, edge_index, edge_attr, batch):

    X, node_mask = to_dense_batch(x, batch)

    

    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

    max_num_nodes = X.size(1)

    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)

    # E = encode_no_edge(E)

    # print(E.shape)
    # print(E[0])

    return PlaceHolder(X=X, E=E, y=None), node_mask



    
