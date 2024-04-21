from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from loss import contrastive_loss
from models.PolyGCL_model import PolyGCL
import torch





def train(model,optim,data):
    model.train()
    optim.zero_grad()

    x = data.x
    x_neg = PolyGCL.get_negative_example(x)
    edge_index = data.edge_index


    pos_Z_L, pos_Z_H = model(x, edge_index)
    neg_Z_L, neg_Z_H = model(x_neg, edge_index)

    g = model.get_global_summary(pos_Z_L, pos_Z_H)

    loss = contrastive_loss(pos_Z_H, neg_Z_H, pos_Z_L, neg_Z_L, g, model.discriminator)
    print(loss.item())

    loss.backward()
    optimizer.step()




if __name__ == '__main__':

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    model = PolyGCL(in_size=dataset.x.shape[-1], hidden_size=128, out_size=128, K=50, dropout_p=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10):
        train(model,optimizer,dataset[0])
