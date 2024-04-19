import torch
import torch.nn as nn
import numpy as np
import math
from torch import nn, Tensor

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1,  max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)

class GCNConv(nn.Module):
    def __init__(self, input_features, output_features, is_self=True):
        super().__init__()

        self.input_features = input_features

        self.output_features = output_features

        self.linear = nn.Linear(input_features, output_features)

        self.is_self = is_self

        if is_self:
            self.s_linear = nn.Linear(input_features, output_features)


    def forward(self, adj_matrix, x):
        neighbor = torch.matmul(adj_matrix, x)

        if self.is_self:
            x = nn.Tanh()(self.s_linear(x)) + nn.Tanh()(self.linear(neighbor))

        else:
            x = nn.Tanh()(self.linear(neighbor))
    
        return x



class ResInf(nn.Module):
    def __init__(self, input_plane, seq_len, trans_layers, gcn_layers, hidden_layers, gcn_emb_size, trans_emb_size, pool_type, n_heads, args, input_size=1):
        super(ResInf, self).__init__()

        self.dim_val = trans_emb_size

        self.args = args

        self.input_plane = input_plane
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if self.input_plane > 1:
            self.sharedMLP = nn.Sequential(
                nn.Conv2d(input_plane, input_plane // 2, 1 ,bias=False), nn.ReLU(),
                nn.Conv2d(input_plane // 2, input_plane, 1, bias=False)
            )

        self.encoder_input_layer = nn.Linear(input_size, self.dim_val)

        self.gcn_layers = gcn_layers

        self.trans_layers = trans_layers

        self.pool_type = pool_type

        self.n_heads = n_heads

        self.dropout_pos_enc = 0.1

        self.seq_len = seq_len

        self.gcn_emb_size = gcn_emb_size

        self.hidden_layers_num = hidden_layers

        self.positional_encoding_layer = PositionalEncoder(d_model=self.dim_val, dropout=0.1, max_seq_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model = self.dim_val, nhead=self.n_heads, batch_first = True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.trans_layers, norm=None)

        self.gcns = nn.ModuleList()

        self.gcns.append(GCNConv(self.dim_val, self.gcn_emb_size))

        for i in range(self.gcn_layers-1):
            self.gcns.append(GCNConv(self.gcn_emb_size, self.gcn_emb_size))

        self.hidden_layers = nn.ModuleList()
        for i in range(self.hidden_layers_num - 1):
            self.hidden_layers.append(nn.Linear(self.gcn_emb_size, self.gcn_emb_size))
            self.hidden_layers.append(nn.Tanh())

        self.resi_net_Linear = nn.Linear(self.gcn_emb_size, self.seq_len)

        if self.hidden_layers_num == 0:
            self.resi_net_down = nn.Linear(self.gcn_emb_size, 1, bias=True)
        else:
            self.resi_net_down = nn.Linear(self.seq_len, 1, bias=True)
        
        self.pred_linear = nn.Linear(1, 1, bias=True)

        

    def forward(self, x, adj_matrix):
        '''
        :param x:  n_realizations * num_nodes + 1 (virtual nodes) * time_ticks
        '''
        feat_len = x.size(2)
        all_nodes_num = x.size(1)
        x = x.reshape(-1, feat_len) 
        x = x.unsqueeze(-1)
        # print(x.shape)
        src = self.encoder_input_layer(x)

        src = self.positional_encoding_layer(src)

        src = self.encoder(src=src) # [n_real * (num_nodes + 1), time_ticks, dim_val]

        embeddings = src[:, -1, :]
        embeddings = embeddings.reshape(-1, all_nodes_num, self.dim_val)

        for i in range(self.gcn_layers):
            embeddings = self.gcns[i](adj_matrix, embeddings)

        if hasattr(self.args, 'att'):
            if self.args.att == True:
                if self.input_plane > 1:
                    avgout = self.sharedMLP(self.avg_pool(embeddings.unsqueeze(0)))
                    maxout = self.sharedMLP(self.max_pool(embeddings.unsqueeze(0)))
                    channel_attention = nn.Sigmoid()((avgout + maxout).squeeze(0))
                    all_node_emb = (embeddings * channel_attention).sum(dim=0)
                else:
                    all_node_emb = embeddings.mean(dim=0)
            else:
                all_node_emb = embeddings.mean(dim=0)
        else:
            if self.input_plane > 1:
                avgout = self.sharedMLP(self.avg_pool(embeddings.unsqueeze(0)))
                maxout = self.sharedMLP(self.max_pool(embeddings.unsqueeze(0)))
                channel_attention = nn.Sigmoid()((avgout + maxout).squeeze(0))
                all_node_emb = (embeddings * channel_attention).sum(dim=0)
            else:
                all_node_emb = embeddings.mean(dim=0)

        if self.pool_type == 'mean':
            res_emb = all_node_emb.mean(dim=0)
        elif self.pool_type == 'max':
            res_emb,_ = all_node_emb.max(dim=0)
        elif self.pool_type == 'virtual':
            res_emb = all_node_emb[-1, :]
        else:
            raise NotImplementedError
        
        true_emb = res_emb.clone()

        if self.hidden_layers_num > 0:
            if self.hidden_layers_num > 1:
                for layer in self.hidden_layers:
                    res_emb = layer(res_emb)
            res_emb = nn.Tanh()(self.resi_net_Linear(res_emb))
        between_down = nn.Tanh()(self.resi_net_down(res_emb))
        resilience = nn.Sigmoid()(self.pred_linear(between_down))

        return resilience, between_down, true_emb

            





        









