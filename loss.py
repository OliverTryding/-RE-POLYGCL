import torch
import torch.nn.functional as F
from models.discriminator import Discriminator

def contrastive_loss(Z_H, 
                     Z_H_tilde, 
                     Z_L, 
                     Z_L_tilde, 
                     g, 
                     D: Discriminator):


    return torch.mean(torch.log(D(g, Z_L)) + torch.log(1 - D(g, Z_L_tilde)) + torch.log(D(g, Z_H)) + torch.log(1 - D(g, Z_H_tilde)))

    