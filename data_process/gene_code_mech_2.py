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
parser.add_argument('--time_tick', type=int, default=400)
parser.add_argument('--type',type=str, default='node')
parser.add_argument('--filename', type=str, default='net1_2', help='File Name')
parser.add_argument('--low', type=int, default=1)
parser.add_argument('--high',type=int, default=30)
parser.add_argument('--per', type=int, default=100)
parser.add_argument('--usenet',type=int,default=1)
parser.add_argument('--mech', type=int, default=1)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--eta', type=float, default=0)
parser.add_argument('--f', type=float, default=1)
parser.add_argument('--h', type=float, default=2)
parser.add_argument('--B', type=float, default=1)
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()
cores = multiprocessing.cpu_count() //2

if args.idx == 0:
    args.f = 1
    args.h = 2
    args.B = 1
elif args.idx == 1:
    args.f = 1
    args.h = 2
    args.B = 2
elif args.idx == 2:
    args.f = 1
    args.h = 3
    args.B = 2.3
elif args.idx == 3:
    args.f = 1
    args.h = 2
    args.B = 2.3
elif args.idx == 4:
    args.f = 1
    args.h = 2
    args.B = 1.7
elif args.idx == 5:
    args.f = 1
    args.h = 3
    args.B = 2.5
elif args.idx == 6:
    args.f = 1
    args.h = 3
    args.B = 2
elif args.idx == 7:
    args.f = 1
    args.h = 2
    args.B = 2.8
elif args.idx == 8:
    args.f = 1
    args.h = 3
    args.B = 2.8
elif args.idx == 9:
    args.f = 1
    args.h = 3
    args.B = 3.5
elif args.idx == 10:
    args.f = 1
    args.h = 2
    args.B = 3.5
elif args.idx == 11:
    args.f = 1
    args.h = 2
    args.B = 4.25
elif args.idx == 12:
    args.f = 1
    args.h = 3
    args.B = 4.7

else:
    raise NotImplementedError


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
    # A = node_removal(remove_num, A)
    A = node_removal_2(remove_num, A)
    num_nodes = len(A)
    idx = torch.nonzero(A).T
    data = A[idx[0], idx[1]]
    A = torch.sparse_coo_tensor(idx, data, A.shape)
    # x0_low = (torch.ones((num_nodes,1)) * 1).to(torch.float32)
    # x0_high = (torch.ones((num_nodes,1)) * 3).to(torch.float32)
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
            numerical_all = ode.odeint(GeneDynamics(A,args.B, args.f, args.h), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            for i in range(int((max_time-1)/2)):
                x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
                numerical_res = ode.odeint(GeneDynamics(A,args.B, args.f, args.h), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
                numerical_all = torch.cat((numerical_all, numerical_res),dim=0) # [n_reali, time_tick, nodes]
            x_0 = (torch.ones((num_nodes,1)) * random.randint(3,10)).to(torch.float32)
            numerical_res = ode.odeint(GeneDynamics(A,args.B, args.f, args.h), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
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
    A = A * 1.5
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
                # numerical_low = ode.odeint(MutualDynamics(A), x0_low, t, method='dopri5')
                # numerical_high = ode.odeint(MutualDynamics(A), x0_high, t, method='dopri5')
                # print(numerical_low[-1].mean().item())
                # print(numerical_high[-1].mean().item())
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
    # x0_low = (torch.ones((num_nodes,1)) * 1).to(torch.float32)
    # x0_high = (torch.ones((num_nodes,1)) * 3).to(torch.float32)
    A = torch.from_numpy(A).to(torch.float32)
    t = torch.linspace(0., args.T, args.time_tick)
    if args.mech == 1:
        with torch.no_grad():
            # x = random.uniform(-2, 1)
            x_0 = (torch.ones((num_nodes,1)) * 0).to(torch.float32)
            numerical_all = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            # for i in range(max_time-1):
            #     x = random.uniform(-2, 1)
            #     x_0 = (torch.ones((num_nodes,1)) * 10**x).to(torch.float32)
            #     numerical_res = ode.odeint(MutualDynamics(A), x_0, t, method='dopri5').unsqueeze(0).squeeze(-1)
            #     numerical_all = torch.cat((numerical_all, numerical_res),dim=0)
                # numerical_low = ode.odeint(MutualDynamics(A), x0_low, t, method='dopri5')
                # numerical_high = ode.odeint(MutualDynamics(A), x0_high, t, method='dopri5')
                # print(numerical_low[-1].mean().item())
                # print(numerical_high[-1].mean().item())
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
    # _,numerical_all,resilience = node_remove_data(args, 2, 2)
    # print(numerical_all.shape)
    # graph_nodes = [10,26,51,41,96,276,87,98,51,25,456,1429,33,88]
    # tt = int( time.time() * 1000.0 )
    # seed = ((tt & 0xff000000) >> 24) +((tt & 0x00ff0000) >>  8) + ((tt & 0x0000ff00) <<  8) + ((tt & 0x000000ff) << 24)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    print(args.f)
    print(args.h)
    print(args.B)
    As = []
    numerical_alls = []
    rs = []
    # for i in tqdm(range(graph_nodes[args.net-1])):
    #     A, numerical_all, r = node_remove_data(args, i, 1)
    #     A = A.cpu().numpy().tolist()
    #     numerical_all = numerical_all.cpu().numpy().tolist()
    #     As.append(A)
    #     numerical_alls.append(numerical_all)
    #     rs.append(r)
    
    # for i in tqdm(range(graph_nodes[args.net+1-1])):
    #     A, numerical_all, r = node_remove_data(args, i, 2)
    #     A = A.cpu().numpy().tolist()
    #     numerical_all = numerical_all.cpu().numpy().tolist()
    #     As.append(A)
    #     numerical_alls.append(numerical_all)
    #     rs.append(r)

##################################################################################
    time1 = time.time()
    print(time1)
    pool = multiprocessing.Pool(cores)
    for i in tqdm(range(args.low, args.high)):
        time4 = time.time()
        if args.type == 'node':
            link_partial = partial(node_remove_data, args, i)
        elif args.type == 'link':
            link_partial = partial(link_remove_data_highk_2, args, i)
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
    # with open('/data/liuchang/rgnn_data/mech_{}/idx_{}/As_{}_{}_{}_{}.json'.format(args.mech, args.idx, use_net, args.low, args.high, args.per), 'w') as f:
    #     json.dump(As,f)
    # with open('/data/liuchang/rgnn_data/mech_{}/idx_{}/numes_{}_{}_{}_{}.json'.format(args.mech, args.idx, use_net, args.low, args.high, args.per), 'w') as f:
    #     json.dump(numerical_alls, f)
    # with open('/data/liuchang/rgnn_data/mech_{}/idx_{}/rss_{}_{}_{}_{}_{:.3f}.json'.format(args.mech, args.idx, use_net, args.low, args.high, args.per, pos_rate), 'w') as f:
    #     json.dump(rs, f)
    # with open('../degree_data/mech_{}/As_{}_{}_{}_{}.json'.format(args.mech, use_net, args.low, args.high, args.per), 'w') as f:
    #     json.dump(As,f)
    # with open('../degree_data/mech_{}/numes_{}_{}_{}_{}.json'.format(args.mech, use_net, args.low, args.high, args.per), 'w') as f:
    #     json.dump(numerical_alls,f)
    # with open('../degree_data/mech_{}/rs_{}_{}_{}_{}_{:.3f}.json'.format(args.mech, use_net, args.low, args.high, args.per, pos_rate), 'w') as f:
    #     json.dump(rs,f)
    with open('../generation_data/mech_{}_new/As_{}_{}_{}_{}_{}.json'.format(args.mech, use_net, args.low, args.high, args.per, args.idx), 'w') as f:
        json.dump(As,f)
    with open('../generation_data/mech_{}_new/numes_{}_{}_{}_{}_{}.json'.format(args.mech, use_net, args.low, args.high, args.per, args.idx), 'w') as f:
        json.dump(numerical_alls,f)
    with open('../generation_data/mech_{}_new/rs_{}_{}_{}_{}_{}_{:.3f}.json'.format(args.mech, use_net, args.low, args.high, args.per,args.idx, pos_rate), 'w') as f:
        json.dump(rs,f)
    print(pos_rate)
    pool.close()
    time2 = time.time()
    print(time2)
    print("Total time:", time2-time1)
##########################################################################################

    # time1 = time.time()
    # print(time1)
    # pool = multiprocessing.Pool(cores)
    # weight_partial = partial(weight_change_data, args, args.usenet)
    # change_rate = []
    # for i in range(438):
    #     change_rate.append(random.random())
    # if args.usenet == 1:
    #     use_net = args.filename + '_A'
    # elif args.usenet == 2:
    #     use_net = args.filename + '_B'
    # else:
    #     raise NotImplementedError
    # results = pool.map(weight_partial, change_rate)
    # for re in results:
    #     A = re[0]
    #     numerical_all = re[1]
    #     r = re[2]
    #     A = A.cpu().numpy().tolist()
    #     numerical_all = numerical_all.cpu().numpy().tolist()
    #     As.append(A)
    #     numerical_alls.append(numerical_all)
    #     rs.append(r)
    # with open('../new_data/{0}/As_{1}.json'.format(args.type, use_net), 'w') as f:
    #     json.dump(As,f)
    # with open('../new_data/{0}/numes_{1}.json'.format(args.type, use_net), 'w') as f:
    #     json.dump(numerical_alls, f)
    # with open('../new_data/{0}/rs_{1}.json'.format(args.type, use_net), 'w') as f:
    #     json.dump(rs, f)
    # pool.close()
    # time2 = time.time()
    # print(time2)
    # print("Total time:", time2-time1)

    # for i in tqdm(range(438)): # net15
    #     time4 = time.time()
    #     if args.type == 'node':
    #         link_partial = partial(node_remove_data, args, i)
    #     elif args.type == 'link':
    #         link_partial = partial(link_remove_data, args, i)
    #     else:
    #         raise NotImplementedError
        
    #     data_list = [args.usenet] * args.per
    #     if args.type == 'node':
    #         if data_list[0] == 1:
    #             use_net = args.filename + '_A'
    #         elif data_list[0] == 2:
    #             use_net = args.filename + '_B'
    #     elif args.type == 'link':
    #         if data_list[0] == 1:
    #             use_net = args.filename + '_B'
    #         elif data_list[0] == 2:
    #             use_net = args.filename + '_A'
    #     else:
    #         raise NotImplementedError
    #     results = pool.map(link_partial, data_list)
        
        # A, numerical_all, r = pool.map(link_partial, data_list)
        # A = A.cpu().numpy().tolist()
        # numerical_all = numerical_all.cpu().numpy().tolist()
        # As.append(A)
        # numerical_alls.append(numerical_all)
        # rs.append(r)
        
        # time3 = time.time()
        # print("1 iter:", time3-time4)
    

    # for i in tqdm(range(1,30)):
    #     for inst in tqdm(range(70)):
    #         A, numerical_all, r = node_remove_data(args, i, 1)
    #         A = A.cpu().numpy().tolist()
    #         numerical_all = numerical_all.cpu().numpy().tolist()
    #         As.append(A)
    #         numerical_alls.append(numerical_all)
    #         rs.append(r)
    #     with open('../new_data/node/As_net18_1_30_100.json', 'w') as f:
    #         json.dump(As,f)
    #     with open('../new_data/node/numes_net18_1_30_100.json', 'w') as f:
    #         json.dump(numerical_alls, f)
    #     with open('../new_data/node/rs_net18_1_30_100.json', 'w') as f:
    #         json.dump(rs, f)
    # normal_data(args)

