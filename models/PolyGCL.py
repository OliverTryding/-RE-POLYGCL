from re import L
from turtle import forward
from typing import Any, Dict, List
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation
import torch.nn as nn
import torch

import matplotlib.pyplot as plt


def get_chebs(x, K):
    """Get Cheb. polynomial value at x for n=0,....,K
    Args:
        x int: value at which to evaluate the polynomials
        K int: maximal order of the polynomials
    Returns:
        torch.Tensor: array containing the values of the polynomials
    """
    chebs = torch.zeros(K)
    chebs[0] = 1
    chebs[1] = x

    # Cheb poly = T_0(x) = 1, T_1(x)= x, T_n+1(x) = 2x T_n(x) - T_{n-1}(X)
    for i in range(2, K):
        chebs[i] = 2*x * chebs[i-1] - chebs[i-2]

    return chebs

def get_cheb_i(x, i):
    """Returns a specific Cheb. polynomial of order i evaluated at x
    Args:
        x int: value at which to evaluate the polynomial
        i int: order
    Returns:
        int: the value of the polynomial
    """
    if i == 0:
        return 1
    if i == 1:
        return x
    return 2*x * get_cheb_i(x,i-1) - get_cheb_i(x, i-2)



class PolyGCL(MessagePassing):
    def __init__(self, K, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.K = K

        self.gammas = nn.Parameter(torch.Tensor(K))
        self.gamma_0 = nn.Parameter(torch.Tensor(1))

    def forward(self, x, edge_index, high_pass=True):

        x_j = torch.Tensor([((j+1)/2)/(self.K+1)*torch.pi for j in range(0,self.K+1)])

        w_k = [] # TODO try avoid the loop

        w_k = torch.Tensor([torch.Tensor([self.gammas[j] * get_cheb_i(x_j[j], k) for j in range(0, self.K+1)]).sum() for k in range(0, self.K+1)])
        w_k *= 2/(self.K+1)
        
        # 'for loop' version, TODO compare execution speed again list comprehesion
        # for k in range(0,self.K+1):
        #     tmp_sum = 0
        #     for j in range(0,self.K+1):
        #         tmp_sum += self.gammas[j] * get_cheb_i(x_j[j], k)
        #     w_k.append(2/(self.K+1)*tmp_sum)

        # TODO get Laplacian, get \hat{L} (approximate \lambda_max as in ChebConv from PyG)
            
        # TODO propagate (mp) for each param [0,..,K]
        

    def message(self, x_j: torch.Tensor, norm) -> torch.Tensor:
        return norm.view(-1,1) * x_j
    



if __name__ == '__main__':
    print('hello world')    

    # Note: x_j have the same range, no matter the K
    plt.scatter([j for j in range(0,100)], torch.Tensor([((j+1)/2)/(100)*torch.pi for j in range(0,100)]))
    plt.scatter([j for j in range(0,10)], torch.Tensor([((j+1)/2)/(10)*torch.pi for j in range(0,10)]))
    plt.scatter([j for j in range(0,50)], torch.Tensor([((j+1)/2)/(50)*torch.pi for j in range(0,50)]))
    plt.show()