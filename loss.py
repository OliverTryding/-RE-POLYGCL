import torch
import torch.nn.functional as F
from models.discriminator import Discriminator


def contrastive_loss(Z_H, Z_H_tilde, Z_L, Z_L_tilde, g, D: Discriminator):
    feat = torch.cat((Z_H, Z_L, Z_H_tilde, Z_L_tilde), dim=0)
    return F.binary_cross_entropy_with_logits(
        D(feat, g),
        torch.cat((torch.ones(Z_H.shape[0]*2, device=g.device), torch.zeros(Z_H.shape[0]*2, device=g.device)))
    )

    