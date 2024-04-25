import torch.nn as nn
import torch
import torch.nn.functional as F


# class Discriminator(nn.Module):
#     def __init__(self, features_dim):
#         super(Discriminator, self).__init__()
#         self.feature_dim = features_dim

#         self.w = nn.Parameter(torch.Tensor(features_dim, features_dim))
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.w)

#     def forward(self, Z, g):
#         return F.sigmoid(Z @ self.w @ g)
    
class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)


    def forward(self, h1, c):
        c_x = c.expand_as(h1).contiguous()
        sc_1 = self.fn(h1, c_x).squeeze(1)
        return sc_1