from sklearn.metrics import hinge_loss
import torch
import torch.nn as nn

from models.ChebNetII import ChebnetII_prop
from models.PolyGCL_layer import PolyGCLLayer
from torch_geometric.nn import BatchNorm
from models.discriminator import Discriminator

class PolyGCL(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, K, dropout_p=0.0, dropout_after=0.0, is_bns = True, act_fn = "relu") -> None:
        super().__init__()

        self.encoder = PolyGCLModel(in_size=in_size, 
                               hidden_size=hidden_size,
                               out_size=out_size,
                               K=K,
                               dropout_p=dropout_p,
                               dropout_after=dropout_after,
                               is_bns=is_bns,
                               act_fn=act_fn)
        

        self.discriminator = Discriminator(out_size)

        # TODO alpha+beta must sum to 1?
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        # self.beta = lambda: 1.0 - self.alpha


    def forward(self, x, edge_index):
        # x can be either a positive or negative (e.g shuffled) example

        # High and Low embeddings
        Z_L = self.encoder(x=x, edge_index=edge_index, high_pass=False)
        Z_H = self.encoder(x=x, edge_index=edge_index, high_pass=True)

        # TODO code applies activation, not mentioned in the paper
        return Z_L, Z_H
    

    def get_negative_example(x):
        # Shuffles along the node dimension (node i gets features of j)
        n = x.shape[-2]
        perm = torch.randperm(n)
        x_ = x.clone()
        return x_[perm]
    
    def get_embedding(self, Z_L, Z_H):
        # a = torch.sigmoid(self.alpha)
        return self.alpha * Z_L + self.beta * Z_H
    
    def get_global_summary(self, Z_L, Z_H):
        return self.get_embedding(Z_L=Z_L, Z_H=Z_H).mean(dim=-2)
    

class PolyGCLModel(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, K, dropout_p=0.0, dropout_after=0.0, is_bns = True, act_fn = "prelu") -> None:
        super().__init__()


        self.input_encoder = nn.Linear(in_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_p) # 0.3
        
        self.convolution = PolyGCLLayer(K)
        # self.convolution = ChebnetII_prop(K)

        self.dropout_after = nn.Dropout(p=dropout_after)  # .2

        # self.norm = torch.nn.BatchNorm1d(hidden_size, momentum=0.01)

        self.up = nn.Sequential(
            torch.nn.BatchNorm1d(hidden_size, momentum=0.01) if is_bns else torch.nn.Identity(),
            nn.Linear(hidden_size, out_size),
            # torch.nn.BatchNorm1d(in_size, momentum=0.01),
            # nn.Linear(in_size, out_size),
            (nn.PReLU() if act_fn == "prelu" else nn.ReLU())
        )


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.input_encoder.weight.data)
        torch.nn.init.xavier_uniform_(self.up[0].weight.data)

    def forward(self, x, edge_index, high_pass=True):
        # encode features to latent dimension
        x = self.input_encoder(x)
        x = self.dropout(x)

        # apply spectral convolution 
        x = self.convolution(x=x, edge_index=edge_index, edge_weights=None, high_pass=high_pass)
        x = self.dropout_after(x)

        # batch norm
        # x = self.norm(x)
        # update the features (UP part of message-passing)
        return self.up(x)