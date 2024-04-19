import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import torchdiffeq as ode
import sys
import functools
import scipy.io as scio
from utils import *
import argparse
from tqdm import tqdm
import json
import random
import multiprocessing
import pickle
from functools import partial
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                    default='euler')  # dopri5
parser.add_argument('--rtol', type=float, default=0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
parser.add_argument('--atol', type=float, default=0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
parser.add_argument('--T', type=float, default=200., help='Terminal Time')
parser.add_argument('--time_tick', type=int, default=100)
parser.add_argument('--type',type=str, default='node')
parser.add_argument('--filename', type=str, default='net1_2', help='File Name')
parser.add_argument('--low', type=int, default=1)
parser.add_argument('--high',type=int, default=30)
parser.add_argument('--per', type=int, default=100)
parser.add_argument('--usenet',type=int,default=1)
parser.add_argument('--mech', type=int, default=1)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--eta', type=float, default=0)
parser.add_argument('--B', type=float, default=0.1)
parser.add_argument('--C', type=float, default=1)
parser.add_argument('--K', type=float, default=5)
parser.add_argument('--D', type=float, default=5)
parser.add_argument('--E', type=float, default=0.9)
parser.add_argument('--H', type=float, default=0.1)
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()
cores = multiprocessing.cpu_count() // 2

if args.idx == 0:
    args.B = 0.1
    args.C = 1
    args.K = 5
    args.D = 5
    args.E = 0.9
    args.H = 0.1
elif args.idx == 1:
    args.B = 0.13
    args.C = 1
    args.K = 5
    args.D = 5
    args.E = 0.9
    args.H = 0.1
elif args.idx == 2:
    args.B = 0.1
    args.C = 1.5
    args.K = 4.5
    args.D = 5
    args.E = 0.9
    args.H = 0.1
elif args.idx == 3:
    args.B = 0.1
    args.C = 1.2
    args.K = 10
    args.D = 5
    args.E = 0.9
    args.H = 0.1  
elif args.idx == 4:
    args.B = 0.1
    args.C = 0.6
    args.K = 8
    args.D = 5
    args.E = 0.9
    args.H = 0.1
elif args.idx == 5:
    args.B = 0.1
    args.C = 1
    args.K = 5
    args.D = 6.5
    args.E = 0.3
    args.H = 1.2
elif args.idx == 6:
    args.B = 0.1
    args.C = 1
    args.K = 5
    args.D = 3.5
    args.E = 0.6
    args.H = 1.8
elif args.idx == 7:
    args.B = 0.1
    args.C = 1
    args.K = 5
    args.D = 5.7
    args.E = 1
    args.H = 0.5
elif args.idx == 8:
    args.B = 0.1
    args.C = 1.5
    args.K = 4.5
    args.D = 3.5
    args.E = 0.6
    args.H = 1.8
elif args.idx == 9:
    args.B = 0.1
    args.C = 0.6
    args.K = 8
    args.D = 6.5
    args.E = 0.3
    args.H = 1.2
elif args.idx == 10:
    args.B = 0.1
    args.C = 1.2
    args.K = 10
    args.D = 5.7
    args.E = 1
    args.H = 0.5


def node_remove_data(args, remove_num, type_net, max_time=10):
    ori_path = '../ori_data/'
    filename = args.filename
    filepath = ori_path + filename + '.mat'
    if type_net == 1:
        A = loadmat(filepath, mat_name='A')
    elif type_net == 2:
        A = loadmat(filepath, mat_name='B')
    else:
        raise NotImplementedError
    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    A = node_removal(remove_num, A)
    num_nodes = len(A)
    idx = torch.nonzero(A).T
    data = A[idx[0], idx[1]]
    A = torch.sparse_coo_tensor(idx, data, A.shape)

    if args.mech == 1:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            
            
            numerical_all = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            
            for i in range(max_time-1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                
                numerical_res = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
              
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            
            
            numerical_res = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 4:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 2:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
            numerical_all = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 3:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_all = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(max_time-1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 3:
                resilience = 0
            else:
                if min > 3.5:
                    resilience = 1
                else:
                    resilience = 0
    elif args.mech == 4:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes, 1)) * random.uniform(0.01, 0.5)).to(torch.float32)
            numerical_all = ode.odeint(SISDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(max_time-1):
                x_0 = (torch.ones((num_nodes, 1)) * random.uniform(0.01, 0.5)).to(torch.float32)
                numerical_res = ode.odeint(SISDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res), dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1
    A = A.to_dense()
    return A, numerical_all, resilience

def link_remove_data_highk(args, remove_num, type_net, max_time=10):
    ori_path = '../ori_data/'
    filename = args.filename
    filepath = ori_path + filename + '.mat'
    A = loadmat(filepath, mat_name='A')
    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    A = link_remove_highk(remove_num, A)
    num_nodes = len(A)
    idx = torch.nonzero(A).T
    data = A[idx[0], idx[1]]
    A = torch.sparse_coo_tensor(idx, data, A.shape)
    if args.mech == 1:

        with torch.no_grad():
            # x = random.uniform(-2, 1)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_all = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(max_time-1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_res = ode.odeint(MutualDynamics(A,args.B, args.K, args.C, args.D, args.E, args.H), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 4:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 2:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 2).to(torch.float32)
            numerical_all = ode.odeint(GeneDynamics(A,1), x_0, t, method='rk4').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,1), x_0, t, method='rk4').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 3:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_all = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2) - 1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 3:
                resilience = 0
            else:
                if min > 3.5:
                    resilience = 1
                else:
                    resilience = 0
    A = A.to_dense()
    return A, numerical_all, resilience

def link_remove_data_highk_2(args, remove_num, type_net, max_time=10):
    ori_path = '../ori_data/'
    filename = args.filename
    filepath = ori_path + filename + '.mat'
    A = loadmat(filepath, mat_name='A')
    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    A = link_remove_highk_2(remove_num, A)
    num_nodes = len(A)
    if args.mech == 1:
        with torch.no_grad():
            x = random.uniform(-2, 1)
            x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
            numerical_all = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(max_time-1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
                
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 4:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 2:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 2).to(torch.float32)
            numerical_all = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 3:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_all = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2) - 1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 3:
                resilience = 0
            else:
                if min > 3.5:
                    resilience = 1
                else:
                    resilience = 0

    return A, numerical_all, resilience

def link_remove_data(args, remove_num, type_net, max_time=10):
    ori_path = '../ori_data/'
    filename = args.filename
    filepath = ori_path + filename + '.mat'
    M = loadmat(filepath, mat_name='M')
    t = torch.linspace(0., args.T, args.time_tick)
    if type_net == 1:
        A = link_removal(remove_num, 1, M)
    elif type_net == 2:
        A = link_removal(remove_num, 2, M)
    num_nodes = len(A)

    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    if args.mech == 1:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_all = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)

            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 4:
                resilience = 0
            else:
                resilience = 1
        
    elif args.mech == 2:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
            numerical_all = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1
    
    elif args.mech == 3:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_all = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2) - 1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 3:
                resilience = 0
            else:
                if min > 3.5:
                    resilience = 1
                else:
                    resilience = 0

    return A, numerical_all, resilience

def weight_change_data(args, type_net, change_rate, max_time=10):
    
    ori_path = '../ori_data/'
    filename = args.filename
    filepath = ori_path + filename + '.mat'
    if type_net == 1:
        A = loadmat(filepath, mat_name='A')
    elif type_net == 2:
        A = loadmat(filepath, mat_name='B')
    else:
        raise NotImplementedError
    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    A = weight_change(change_rate, A)
    num_nodes = len(A)

    if args.mech == 1:
        with torch.no_grad():
            x = random.uniform(-2, 1)
            x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
            numerical_all = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(max_time-1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
                # numerical_low = ode.odeint(MutualDynamics(A), x0_low, t, method='dopri5')
                # numerical_high = ode.odeint(MutualDynamics(A), x0_high, t, method='dopri5')
                # print(numerical_low[-1].mean().item())
                # print(numerical_high[-1].mean().item())
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 4:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 2:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
            numerical_all = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,1), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            numerical_all_mean = numerical_all.mean(dim=2)
            final_mean = numerical_all_mean[:,-1].mean()
            if final_mean < 1e-5:
                resilience = 0
            else:
                resilience = 1

    elif args.mech == 3:
        with torch.no_grad():
            x_0 = (torch.ones((num_nodes,1)) * 5).to(torch.float32)
            numerical_all = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2) - 1):
                x = random.uniform(-2, 1)
                x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
                numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_res = ode.odeint(NeuronalDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
            numerical_all_mean = numerical_all.mean(dim=2)
            max = numerical_all_mean.max(dim=0)[0][-1].item()
            min = numerical_all_mean.min(dim=0)[0][-1].item()
            if abs(max - min) > 3:
                resilience = 0
            else:
                if min > 3.5:
                    resilience = 1
                else:
                    resilience = 0
    
    return A, numerical_all, resilience

if __name__ == '__main__':
 

    As = []
    numerical_alls = []
    rs = []

##################################################################################
    
    flag = True

    while flag:
        time1 = time.time()
        print(time1)
        pool = multiprocessing.Pool(cores)
        for i in tqdm(range(args.low, args.high)):
            time4 = time.time()
            if args.type == 'node':
                link_partial = partial(node_remove_data, args, i)
            elif args.type == 'link':
                link_partial = partial(link_remove_data_highk, args, i)
                # link_partial = partial(link_remove_data, args, i)
            else:
                raise NotImplementedError
            data_list = [args.usenet] * args.per
            if args.type == 'node':
                if data_list[0] == 1:
                    use_net = args.filename + '_A'
                elif data_list[0] == 2:
                    use_net = args.filename + '_B'
            elif args.type == 'link':
                if data_list[0] == 1:
                    use_net = args.filename + '_A'
                elif data_list[0] == 2:
                    use_net = args.filename + '_B'
            else:
                raise NotImplementedError
            results = pool.map(link_partial, data_list)
            for re in results:
                A = re[0]
                numerical_all = re[1]
                r = re[2]
                A = A.cpu().numpy().tolist()
                numerical_all = numerical_all.cpu().numpy().tolist()
                As.append(A)
                numerical_alls.append(numerical_all)
                rs.append(r)
            # A, numerical_all, r = pool.map(link_partial, data_list)
            # A = A.cpu().numpy().tolist()
            # numerical_all = numerical_all.cpu().numpy().tolist()
            # As.append(A)
            # numerical_alls.append(numerical_all)
            # rs.append(r)
            
            time3 = time.time()
            print("1 iter:", time3-time4)
        p = np.array(rs)
        pos_rate = float(p.sum()/p.shape)

        if pos_rate >= 0.45 and pos_rate <= 0.55:

            flag = False
        else:
            if pos_rate < 0.45:
                args.low = args.low - 2
                args.high = args.high - 2
            else:
                args.low = args.low + 2
                args.high = args.high + 2

        pool.close()
    if not os.path.exist('../generation_data_new/mech_{}_{}/'.format(args.mech, args.type)):
        os.mkdir('../generation_data_new/mech_{}_{}/'.format(args.mech, args.type))
        
    with open('../generation_data_new/mech_{}_{}/As_{}_{}_{}_{}_{}.pkl'.format(args.mech, args.type, use_net, args.low, args.high, args.per, args.idx), 'wb') as f:
        pickle.dump(As,f)
    with open('../generation_data_new/mech_{}_{}/numes_{}_{}_{}_{}_{}.pkl'.format(args.mech, args.type, use_net, args.low, args.high, args.per, args.idx), 'wb') as f:
        pickle.dump(numerical_alls,f)
    with open('../generation_data_new/mech_{}_{}/rs_{}_{}_{}_{}_{}_{:.3f}.pkl'.format(args.mech, args.type, use_net, args.low, args.high, args.per,args.idx, pos_rate), 'wb') as f:
        pickle.dump(rs,f)

    # wit
    print(pos_rate)
    
    time2 = time.time()
    print(time2)
    print("Total time:", time2-time1)

