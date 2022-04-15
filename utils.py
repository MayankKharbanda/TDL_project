import torch
import numpy as np
import scipy.sparse as sp


def is_sparse_tensor(tensor):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False



def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize_adj_tensor(adj, sparse=False):
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx)
    else:
        mx = adj + torch.eye(adj.shape[0])
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

