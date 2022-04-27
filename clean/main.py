import torch
from gcn import *
import pickle
import numpy as np


def get_train_val_test_gcn(labels, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: int((0.10*len(labels))/nclass)])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[int((0.10*len(labels))/nclass): ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: int(0.10*len(labels))]
    idx_test = idx_unlabeled[int(0.10*len(labels)):]
    return idx_train, idx_val, idx_test



f = open("../cora_graph", "rb")
graph = pickle.load(f)
adj = torch.from_numpy(graph['adj_matrix']); features = torch.from_numpy(graph['features']); labels = torch.from_numpy(graph['labels'])
adj = torch.ceil(adj).float(); features = torch.ceil(features).float()

torch.manual_seed(1)
idx_train, idx_val, idx_test = get_train_val_test_gcn(labels, seed = 1)
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0.5, weight_decay=5e-4)
surrogate.fit(features, adj, labels, idx_train, idx_val)
surrogate.test(idx_test)
