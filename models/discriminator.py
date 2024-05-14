import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h, c):
        c_x = c.expand_as(h).contiguous()
        sc = self.fn(h, c_x).squeeze(1)
        return sc