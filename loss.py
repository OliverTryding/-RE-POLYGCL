import torch
import torch.nn.functional as F
from models.discriminator import Discriminator

def contrastive_loss(Z_H, 
                     Z_H_tilde, 
                     Z_L, 
                     Z_L_tilde, 
                     g, 
                     D: Discriminator):


    return -torch.mean(torch.log(D(Z_L,g)) + torch.log(1 - D(Z_L_tilde,g)) + torch.log(D(Z_H,g)) + torch.log(1 - D(Z_H_tilde,g)))

    