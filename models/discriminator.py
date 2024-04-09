from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, features_dim):
        super(Discriminator, self).__init__()
        self.feature_dim = features_dim

        self.w = nn.Parameter(torch.Tensor(features_dim, features_dim))

    def forward(self, g, Z):
        return F.sigmoid(Z @ self.w @ g.mT)
    


if __name__ == '__main__':

    g = torch.rand(1,32)

    Z = torch.rand(120, 32)

    d = Discriminator(32)

    out = d(g, Z)

    print(out.shape)