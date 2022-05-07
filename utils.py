import torch
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets

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


def edge_vector_to_adj_matrix(num_vertices, edge_vector):
    adj_matrix = np.zeros((num_vertices, num_vertices))
    for i in range(edge_vector[0].shape[0]):
        start = edge_vector[0][i]
        end = edge_vector[1][i]
        adj_matrix[start, end] = 1
    return adj_matrix

def get_dataset(dataset_name):
    if(dataset_name == "KarateClub"):
        dataset = torch_geometric.datasets.KarateClub().data
    
    elif(dataset_name == "CiteSeer"):
        dataset = torch_geometric.datasets.Planetoid(dataset_name, dataset_name).data

    elif(dataset_name == "Cora"):
        dataset = torch_geometric.datasets.Planetoid(dataset_name, dataset_name).data

    else:
        return None
    
    edge_vector = dataset.edge_index.numpy()

    adj_matrix = edge_vector_to_adj_matrix(dataset.x.shape[0], edge_vector)
    adj_matrix = torch.from_numpy(adj_matrix)
    feature_matrix = dataset.x
    labels = dataset.y

    return [adj_matrix, feature_matrix, labels]
