from turtle import forward
from sklearn.metrics import hinge_loss
import torch
import torch.nn as nn

from models.PolyGCL_layer import PolyGCLLayer
from torch_geometric.nn import BatchNorm
from discriminator import Discriminator

class PolyGCL(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, K, dropout_p=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = PolyGCLModel(in_size=in_size, 
                               hidden_size=hidden_size,
                               out_size=out_size,
                               K=K,
                               dropout_p=dropout_p)
        

        self.discriminator = Discriminator(out_size)

        # TODO alpha+beta must sum to 1?
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))


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
        return self.alpha * Z_L + self.beta * Z_H
    
    def get_global_summary(self, Z_L, Z_H):
        return self.get_embedding(Z_L=Z_L, Z_H=Z_H).mean(dim=-2)
    

class PolyGCLModel(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, K, dropout_p=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



        self.input_encoder = nn.Linear(in_size, hidden_size)

        self.convolution = PolyGCLLayer(K)

        self.dropout = nn.Dropout(p=dropout_p)

        self.norm = BatchNorm(in_channels=hidden_size)

        self.up = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.PReLU()
        )




    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.input_encoder.weight.data)
        torch.nn.init.xavier_uniform_(self.up[0].weight.data)

    def forward(self, x, edge_index, high_pass=True):
        # encode features to latent dimension
        x = self.input_encoder(x)
        x = self.dropout(x)

        # apply spectral convolution 
        out = self.convolution(x=x, edge_index=edge_index, edge_weights=None, high_pass=high_pass)
        # batch norm
        out = self.norm(out)
        # update the features (UP part of message-passing)
        return self.up(out)