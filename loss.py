import torch
import torch.nn.functional as F
from models.discriminator import Discriminator

def contrastive_loss(Z_H, 
                     Z_H_tilde, 
                     Z_L, 
                     Z_L_tilde, 
                     g, 
                     D: Discriminator):

    eps = 1e-15
    return -torch.mean(torch.log(D(Z_L,g)+eps) + torch.log(1 - D(Z_L_tilde,g)+eps) + torch.log(D(Z_H,g)+eps) + torch.log(1 - D(Z_H_tilde,g)+eps))

    