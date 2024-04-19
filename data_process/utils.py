from os import remove
import numpy as np
import pandas as pd
import torchdiffeq as ode
import torch
import torch.nn as nn
import scipy.io as scio
import networkx as nx
import random

def list_to_array (x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(x)],axis=1)
    return dff.fillna(0).values.T.astype(int)

def bfs(A, c):
    '''
    A: ndarray adjacent matrix
    c: node

    return vis set 
    '''
    vis = set([c])
    new = set([c])
    while True:
        if len(new) == 0:
            break
        else:
            tem = set()
            new = list(new)
            for i in range(len(new)):
                tem0 = set(np.where(A[new[i], :] != 0)[0].tolist())
                tem = tem.union(tem0)
            new = tem - vis
            vis = vis.union(vis, tem)
    return vis

def find_gaint_component(A):
    '''
    A: ndarray adjacent matrix

    return ndarray cluster 
    '''
    n = len(A)
    all = np.arange(n)
    unvis = all.copy()
    num = 0
    cluster = []
    while len(unvis) != 0:
        num = num + 1
        c = unvis[0]
        vis = list(bfs(A, c))
        cluster.append([len(vis)] + vis)
        vis = set(vis)
        unvis = np.array(list(set(unvis) - set(vis)))
    cluster = list_to_array(cluster)
    mm, nn = cluster.shape
    if mm == 1:
        cluster = list(cluster[0][1:])
    else:
        for i in range(mm):
            if cluster[i, nn-1] != 0:
                row = i
                break
        cluster = cluster[row, 1:]
    return cluster

def iteration_real_M(odefunc, A, vt, x0):
    '''
    :param A: ndarray
    '''
    n = len(A)
    out = ode.odeint(odefunc, x0, vt)
    return out

class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, b=0.1, k=5., c=1., d=5., e=0.9, h=0.1):
        super(MutualDynamics, self).__init__()
        self.A = A   # Adjacency matrix, symmetric
        self.b = b
        self.k = k
        self.c = c
        self.d = d
        self.e = e
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x/self.k) * (x/self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                outer = torch.sparse.mm(self.A,
                                        torch.mm(x, x.t()) / (self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            else:
                outer = torch.mm(self.A,
                                    torch.mm(x, x.t()) / (
                                                self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            f += torch.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += self.A[i, j]*(x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
        return f

class NeuronalDynamics(nn.Module):
    def __init__(self, A, u=3.5, d=2):
        super(NeuronalDynamics, self).__init__()
        self.A = A
        self.u = u
        self.d = d
    
    def forward(self, t, x):
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -x + torch.sparse.mm(self.A, 1 / (1 + torch.exp(self.u - self.d*x)))
        else:
            f = -x + torch.mm(self.A, 1/(1 + torch.exp(self.u - self.d*x)))
        return f
        

class GeneDynamics(nn.Module):
    def __init__(self,  A,  b, f=1, h=2):
        super(GeneDynamics, self).__init__()
        self.A = A   # Adjacency matrix
        self.b = b
        self.f = f
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -self.b * (x ** self.f) + torch.sparse.mm(self.A, x**self.h / (x**self.h + 1))
        else:
            f = -self.b * (x ** self.f) + torch.mm(self.A, x ** self.h / (x ** self.h + 1))
        return f


class SISDynamics(nn.Module):
    def __init__(self, A, d=6):
        super(SISDynamics, self).__init__()
        self.A = A
        self.d = d
        # self.u = u
        # self.d = d
    def forward(self, t, x):
        
        f = -self.d * x
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            # f = -x + torch.sparse.mm(self.A, 1 / (1 + torch.exp(self.u - self.d*x)))
            # outer = torch.sparse.mm(self.A, torch.mm(1-x, x.t()))
            f += torch.mul(1-x, torch.sparse.mm(self.A, x))
        else:
            f += torch.mul(1-x, torch.mm(self.A, x))
            # outer = torch.mm(self.A, torch.mm(1-x, x.t()))
            # f = -x + torch.mm(self.A, 1/(1 + torch.exp(self.u - self.d*x)))
        # f += torch.diag(outer).view(-1,1)
        return f

def weight_change_2(rate, A):
    A = A.numpy()
    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    components = nx.strongly_connected_components(G)
    lii = [c for c in components]
    cluster = []
    for sample in lii:
        if len(sample) > 1:
            for com in sample:
                cluster.append(com)
    A = A[cluster][:, cluster]
    n = len(A)
    A = A * 2
    if rate > 1:
        raise NotImplementedError
    A = torch.from_numpy(A).to(torch.float32)
    A0 = torch.rand(n,n)
    A0 = A0 * A
    A0 = A0*(torch.sum(torch.sum(A)))/(torch.sum(torch.sum(A0)))
    A = rate * A0
    return A

def weight_change(rate, A):
    A = A.numpy()
    G = nx.from_numpy_array(A)
    cluster = max(nx.connected_components(G), key=len)
    cluster = list(cluster)
    # cluster = find_gaint_component(A)
    A = A[cluster][:, cluster]
    n = len(A)
    A = A * 2
    if rate > 1:
        raise NotImplementedError
    A = torch.from_numpy(A).to(torch.float32)
    A0 = torch.rand(n,n)
    A0 = A0 * A
    A0 = A0*(torch.sum(torch.sum(A)))/(torch.sum(torch.sum(A0)))
    A = rate * A0
    return A

def node_removal(remove_num, A):
    n = len(A)
    check = False
    while not check:
        # all = torch.randperm(n)
        if remove_num > n:
            raise NotImplementedError
        reserve = random.sample(list(range(n)), n-remove_num)
        A1 = A[reserve][:, reserve].clone()
        # A = A[all].T[all].T
        # num_remove = remove_num
        # A = A[num_remove:, num_remove:]
        A1 = A1.numpy()
        G = nx.from_numpy_array(A1)
        cluster = max(nx.connected_components(G), key=len)
        cluster = list(cluster)
        # cluster = find_gaint_component(A)

        A1 = A1[cluster][:, cluster]
        check = np.any(np.array(A1)) and (len(A1) > 1)
    A = torch.from_numpy(A1).to(torch.float32)
    return A

def node_removal_2(remove_num, A):
    A = A.numpy()
    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    components = nx.strongly_connected_components(G)
    lii = [c for c in components]
    cluster = []
    for sample in lii:
        if len(sample) > 1:
            for com in sample:
                cluster.append(com)
    A = A[cluster][:, cluster]
    n = len(A)
    if remove_num > n:
        raise NotImplementedError
    reserve = random.sample(list(range(n)), n-remove_num)
    A = A[reserve][:, reserve]
    A = torch.from_numpy(A).to(torch.float32)
    return A

def link_remove_highk(remove_num, A):
    A = A.numpy()
    check = False
    while not check:
        G = nx.from_numpy_array(A, create_using=nx.Graph())
        edges = list(G.edges)
        chosen_edges = random.sample(edges, k=remove_num)
        for chosen_edge in chosen_edges:
            G.remove_edge(chosen_edge[0], chosen_edge[1])
        cluster = max(nx.connected_components(G), key=len)
        cluster = list(cluster)
        A1 = nx.to_numpy_array(G)
        A1 = A1[cluster][:, cluster]
        check = np.any(np.array(A1)) and (len(A1) > 1)
    A = torch.from_numpy(A1).to(torch.float32)
    return A

def link_remove_highk_2(remove_num, A):
    A = A.numpy()
    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    components = nx.strongly_connected_components(G)
    lii = [c for c in components]
    cluster = []
    for sample in lii:
        if len(sample) > 1:
            for com in sample:
                cluster.append(com)
    A = A[cluster][:, cluster]
    check = False
    while not check:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph())
        edges = list(G.edges)
        chosen_edges = random.sample(edges, k=remove_num)
        for chosen_edge in chosen_edges:
            G.remove_edge(chosen_edge[0], chosen_edge[1])
        components = nx.strongly_connected_components(G)
        lii = [c for c in components]
        cluster = []
        for sample in lii:
            if len(sample) > 1:
                for com in sample:
                    cluster.append(com)
        A1 = nx.to_numpy_array(G)
        A1 = A1[cluster][:, cluster]
        check = np.any(np.array(A1))
    A = torch.from_numpy(A1).to(torch.float32)
    return A

def loadmat(file_path, mat_name):
    data = scio.loadmat(file_path)
    return data[mat_name]

def link_removal(remove_num, type_net, M):
    m, n = M.shape
    if type_net == 1 and remove_num >= m:
        raise NotImplementedError
    if type_net == 2 and remove_num >= n:
        raise NotImplementedError
    num_remove = remove_num
    A = remove_one_effect_the_other(M, type_net, num_remove)
    return A

def remove_one_effect_the_other(M, type_net, num_remove):
    m, n = M.shape
    if type_net == 1:
        reserve = random.sample(list(range(m)), m-num_remove)
        M = M[reserve]
        m,n = M.shape
        A = np.zeros((n,n))
        kn = np.sum(M, axis=1)
        for i in range(m):
            for j in range(n-1):
                for k in range(j+1, n):
                    if M[i,j] != 0 and M[i, k] != 0:
                        A[j,k] = A[j,k] + M[i, k]/kn[i] + M[i, j]/kn[i]
                        A[k,j] = A[k,j] + M[i, j]/kn[i] + M[i, k]/kn[i]
    else:
        reserve = random.sample(list(range(n)), n-num_remove)
        M = M[:, reserve]
        m,n = M.shape
        A = np.zeros((m,m))
        km = np.sum(M, axis=0)
        for i in range(n):
            for j in range(m-1):
                for k in range(j+1, m):
                    if M[j, i] != 0 and M[k, i] != 0:
                        A[j,k] = A[j,k]+M[k,i]/km[i] + M[j, i]/km[i]
                        A[k,j] = A[k,j]+M[j,i]/km[i] + M[k, i]/km[i]
        
    # A = A.numpy()
    G = nx.from_numpy_array(A)
    cluster = max(nx.connected_components(G), key=len)
    cluster = list(cluster)
    A = A[cluster][:, cluster]
    return A
