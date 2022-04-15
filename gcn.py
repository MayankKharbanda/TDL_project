import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import utils
from copy import deepcopy
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
    ## Dict. Learning ##
####################################

class DictLearn(nn.Module):
    def __init__(self,n,m):
        super(DictLearn, self).__init__()

        self.W = nn.Parameter(torch.randn(n, m, requires_grad=False))
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
    def forward(self, Y, SC, K):
        
        # normalizing Dict
        self.W.requires_grad_(False)
        self.W.data = NormDict(self.W.data)
        
        # Sparse Coding
        if SC == 'IHT':
            Gamma,residual, errIHT = IHT(Y,self.W,K)
        elif SC == 'fista':
            Gamma,residual, errIHT = FISTA(Y,self.W,K)
        else: print("Oops!")
        
        
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(self.W,Gamma)
        return X, Gamma, errIHT

        

def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device)
    Gamma = Gamma * mask
    return Gamma#, mask.data.nonzero()


def soft_threshold(X, lamda):
    Gamma = X.clone()
    Gamma = torch.sign(Gamma) * F.relu(torch.abs(Gamma)-lamda)
    return Gamma.to(device)



def IHT(Y,W,K):
    
    c = PowerMethod(W)
    eta = 1/c
    Gamma = hard_threshold_k(torch.mm(Y,eta*W),K)    
    residual = torch.mm(Gamma, W.transpose(1,0)) - Y
    IHT_ITER = 50
    
    norms = np.zeros((IHT_ITER,))

    for i in range(IHT_ITER):
        Gamma = hard_threshold_k(Gamma - eta * torch.mm(residual, W), K)
        residual = torch.mm(Gamma, W.transpose(1,0)) - Y
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------


def FISTA(Y,W,lamda):
    
    c = PowerMethod(W)
    eta = 1/c
    FISTA_ITER = 20
    norms = np.zeros((FISTA_ITER,))
    # print(c)
    # plt.spy(Gamma); plt.show()
    # pdb.set_trace()
    
    Gamma = soft_threshold(torch.mm((eta*W).transpose(1,0),Y),lamda)
    Z = Gamma.clone()
    Gamma_1 = Gamma.clone()
    t = 1
    
    for i in range(FISTA_ITER):
        Gamma_1 = Gamma.clone()
        residual = torch.mm(W,Z) - Y
        Gamma = soft_threshold(Z - eta * torch.mm(W.transpose(1,0),residual), lamda/c)
        
        t_1 = t
        t = (1+np.sqrt(1 + 4*t**2))/2
        #pdb.set_trace()
        Z = Gamma + ((t_1 - 1)/t * (Gamma - Gamma_1)).to(device)
        
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).to(device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm
    
    return nm

#--------------------------------------------------------------


def showFilters(W,ncol,nrows):
    p = int(np.sqrt(W.shape[0]))+2
    Nimages = W.shape[1]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[:,indx].reshape(p-2,p-2)
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im,(1,1),mode='constant')
            indx += 1
            
    return Mosaic

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True):

        super(GCN, self).__init__()
        self.dl1 = DictLearn(2708,int(nfeat/2))
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        
    def forward(self, x, adj,SC,K):
        x_dec,gamma,errIHT = self.dl1(x,SC,K)
        if self.with_relu:
            x = F.relu(self.gc1(x_dec, adj))
        else:
            x = self.gc1(x_dec, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1),x_dec,gamma,errIHT

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, SC, K, idx_train, idx_val=None, train_iters=20, initialize=False, normalize=True):
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        return self._train_with_val(labels, idx_train, idx_val, train_iters,SC,K)


    def _train_with_val(self, labels, idx_train, idx_val, train_iters,SC,K):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output,x_dec,gamma,errIHT = self.forward(self.features, self.adj_norm,SC,K)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_sparse = F.mse_loss(x_dec,self.features)
            loss_total = loss_train+loss_sparse
            print(loss_total)
            loss_total.backward()
            optimizer.step()

            self.eval()
            output,x_dec,gamma,errIHT = self.forward(self.features, self.adj_norm,SC,K)
            loss_v1 = F.nll_loss(output[idx_val], labels[idx_val])
            loss_sparse = F.mse_loss(x_dec,self.features)
            loss_val = loss_v1+loss_sparse

            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        self.load_state_dict(weights)
        return loss_train.item()



    def test(self, idx_test,SC,K):
        self.eval()
        output,x_dec,gamma,errIHT = self.predict(SC,K)

        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self,SC,K):
        self.eval()
        return self.forward(self.features, self.adj_norm,SC,K)
