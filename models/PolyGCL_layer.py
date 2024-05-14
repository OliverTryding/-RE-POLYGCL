from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian
import torch.nn as nn
import torch
import torch.nn.functional as F


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
        chebs[i] = 2 * x * chebs[i - 1] - chebs[i - 2]

    return chebs


def get_cheb_i(x, i):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def prefix_sum(gammas, gamma_0):
    gammas_H = torch.concatenate([gamma_0, gammas], dim=0)

    return torch.cumsum(gammas_H, dim=0)


def prefix_diff(gammas, gamma_0):
    gammas_L = torch.concatenate([gamma_0, -gammas], dim=0)

    return torch.cumsum(gammas_L, dim=0)


class PolyGCLLayer(MessagePassing):
    def __init__(self, K, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.K = K

        self.gamma_0_L = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.gamma_0_H = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.gammas_L = nn.Parameter(torch.Tensor(self.K), requires_grad=True)
        self.gammas_H = nn.Parameter(torch.Tensor(self.K), requires_grad=True)

        self._reset_parameter()

    def _reset_parameter(self):
        self.gammas_H.data.fill_(2.0 / self.K)
        self.gammas_L.data.fill_(2.0 / self.K)
        self.gamma_0_L.data.fill_(2.0)
        self.gamma_0_H.data.fill_(0.0)

    def _get_norm(self, edge_index, num_nodes, edge_weights=None, lambda_max=2.0):

        # Get normalized laplacian L
        edge_index_lap, edge_weights_lap = get_laplacian(edge_index=edge_index,
                                                         edge_weight=edge_weights,
                                                         normalization='sym',
                                                         num_nodes=num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weights_lap.max()

        # \tilde{L} = 2L/\lambda_max - I_n
        edge_weights_lap = (2 * edge_weights_lap / lambda_max)
        edge_index_lap.masked_fill(edge_index_lap == float('inf'), 0)

        self_loops = edge_index_lap[0] == edge_index_lap[1]
        assert self_loops.float().sum() == num_nodes, f"There should be a self-loop for each node. {self_loops.float().sum()} vs. {num_nodes}"

        edge_weights_lap[self_loops] -= 1

        return edge_index_lap, edge_weights_lap

    def forward(self, x, edge_index, edge_weights, high_pass=True):
        # Inspration from ChebConv
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html##ChebConv

        x_j = torch.Tensor([(j + 0.5) / (self.K + 1) * torch.pi for j in range(0, self.K + 1)][::-1])
        x_j = torch.cos(x_j)

        prefixed_gammas = None
        if high_pass:
            prefixed_gammas = prefix_sum(gammas=F.relu(self.gammas_H),
                                         gamma_0=self.gamma_0_H)  # TODO, if two sets of gammas are introduced, change here
        else:
            prefixed_gammas = prefix_diff(gammas=F.relu(self.gammas_L), gamma_0=self.gamma_0_L)
            prefixed_gammas = F.relu(prefixed_gammas)

        ws = prefixed_gammas.clone()
        for k in range(0, self.K + 1):
            ws[k] = prefixed_gammas[0] * get_cheb_i(x_j[0], k)
            for j in range(1, self.K + 1):
                ws[k] = ws[k] + prefixed_gammas[j] * get_cheb_i(x_j[j], k)
            ws[k] = 2 * ws[k] / (self.K + 1)

        edge_index_lap, edge_weights_lap = self._get_norm(edge_index=edge_index,
                                                          num_nodes=x.shape[0],
                                                          edge_weights=edge_weights,
                                                          lambda_max=2)

        T_0x = x
        T_1x = self.propagate(edge_index=edge_index_lap, x=x, norm=edge_weights_lap)

        out = ws[0] / 2 * T_0x + ws[1] * T_1x

        for k in range(2, self.K + 1):
            T_2x = self.propagate(edge_index=edge_index_lap, x=T_1x, norm=edge_weights_lap)
            T_2x = 2. * T_2x - T_0x

            out = out + ws[k] * T_2x

            T_0x, T_1x = T_1x, T_2x

        return out

    def message(self, x_j: torch.Tensor, norm) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


if __name__ == '__main__':
    net = PolyGCLLayer(10)

    edge_index = torch.Tensor([[0, 1, 1, 2], [1, 0, 2, 1]]).to(torch.int64)
    x = torch.Tensor([[42], [70], [65]])

    out = net(x=x, edge_index=edge_index, edge_weights=None, high_pass=False)
    print(out)
