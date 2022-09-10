#!/usr/bin/env python3
from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.nn.modules.module import Module
import scipy.sparse as sp

from torch.utils.data import Dataset
import os
import numpy


# torch.cuda.set_device(1)


# Create logger
def eva(y_true, y_pred):
    ari = ari_score(y_true, y_pred)
    return ari
def load_graph(n, prefix, output_path):
    Ag = output_path + prefix + "_ag.txt"
    Pe = output_path + prefix + "_pe.txt"
    Seq = output_path + prefix + "_seq.txt"
    edges_ag = np.genfromtxt(Ag, dtype=np.int32)
    edges_pe = np.genfromtxt(Pe, dtype=np.int32)
    edges_seq = np.genfromtxt(Seq, dtype=np.int32)

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.vstack((edges_ag,edges_seq))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class load_data(Dataset):
    def __init__(self, prefix, output_path,n_samples):
        x1 = np.loadtxt('{}/{}_normalized_contig_tetramers.txt'.format(output_path, prefix), dtype=float)
        if n_samples != 0:
           x2 = np.loadtxt('{}/{}_normalized_coverages.txt'.format(output_path, prefix), dtype=float)
           if n_samples==1:
              mi=min(x2)
              mami=max(x2)-mi
              x2 =(x2-mi)/mami
              x2=x2.reshape((len(x2),1))           
           else:       
               for i in range(len(x2)):
                   if (max(x2[i])-min(x2[i]))==0:
                       x2[i]=(x2[i]-x2[i])
                   else:
                       x2[i] =(x2[i]-min(x2[i]))/(max(x2[i])-min(x2[i]))
           self.x=np.hstack((x1,x2))
        else:
           self.x=x1  
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, enc_h1, enc_h2, z

class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                n_input, n_z, n_clusters, pretrain_path, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_z)
        self.gnn_4 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(n_h1, n_h2, n_z, cluster_K, n_input, prefix, output_path,learn_rate,length_list,cpu_num, n_samples,node):
    device = torch.device("cpu")
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    cluster_K = int(cluster_K)
    dataset = load_data(prefix, output_path, n_samples)
    n_contigs = dataset.__len__()
    model = SDCN(n_h1, n_h2, n_h2, n_h1,
                n_input=n_input,
                n_z=n_z,
                n_clusters=cluster_K,
                pretrain_path=output_path + prefix + ".pkl",
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=learn_rate)
    adj = load_graph(n_contigs, prefix, output_path)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)

    with torch.no_grad():
        _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=cluster_K, n_init=30)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)


    lossmin = 999
    lossmin2 =999
    K = range(300)
    Kmin=0
    Kmin2=0
    for epoch in K:
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)

        x_bar, q, pred, _ = model(data, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)


        loss =0.1*kl_loss+0.1*ce_loss+10*re_loss


        print('{} loss: {}'.format(epoch, loss))

        if loss<lossmin:
           lossmin=loss
           Kbest = epoch
           result=res2
           ypred=pred
        if ce_loss<lossmin2:
           lossmin2=ce_loss
           Kbest2 = epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(Kbest,Kbest2)
    return result, Kbest,ypred



