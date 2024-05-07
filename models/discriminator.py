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

# class Discriminator(nn.Module):
#     def __init__(self, dim):
#         super(Discriminator, self).__init__()
#         self.fn = nn.Bilinear(dim, dim, 1)


#     def forward(self, h1, h2, h3, h4, c):
#         c_x = c.expand_as(h1).contiguous()
#         # dim = 10
#         # (batch_size, dim) (dim, dim) (batch_size | 1, dim)
#         # positive
#         sc_1 = self.fn(h2, c_x).squeeze(1)
#         sc_2 = self.fn(h1, c_x).squeeze(1)

#         # negative
#         sc_3 = self.fn(h4, c_x).squeeze(1)
#         sc_4 = self.fn(h3, c_x).squeeze(1)

#         logits = torch.cat((sc_1, sc_2, sc_3, sc_4))

#         return logits