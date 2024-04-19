#from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, features_dim):
        super(Discriminator, self).__init__()
        self.feature_dim = features_dim

        self.w = nn.Parameter(torch.Tensor(features_dim, features_dim))

    def forward(self, Z, g):
        return F.sigmoid(Z @ self.w @ g.mT)
    
