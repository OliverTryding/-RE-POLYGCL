from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from loss import contrastive_loss
from models.PolyGCL_model import PolyGCL
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime




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

    loss.backward()
    optim.step()

    return loss.item()


def get_masks(n_nodes, train_split=0.8, val_split=0.1):
    train_n = int(train_split * n_nodes)
    val_n = int(val_split * n_nodes)
    test_n = n_nodes - train_n - val_n

    permutation = torch.randperm(n_nodes)

    train_nodes = permutation[:train_n]
    val_nodes = permutation[train_n:train_n+val_n]
    test_nodes = permutation[train_n+val_n:]

    return train_nodes, val_nodes, test_nodes




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    model = PolyGCL(in_size=dataset.x.shape[-1], hidden_size=128, out_size=128, K=10, dropout_p=0.4)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    
    run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

    writer = SummaryWriter(log_dir=f'runs/{run_name}')


    data = dataset[0].to(device)

    # TODO add early stopper
    # TODO add validation and test mask -> compute loss only on train set
    # TODO add evaluation step

    for i in range(5000):
        loss = train(model,optimizer,data)
        print(loss)

        print(model.encoder.convolution.gammas_L, "low gammas")
        print(model.encoder.convolution.gammas_H, "high gammas")

        writer.add_scalar('Loss/train', loss, i)
        writer.add_scalar('beta/train', model.beta, i)
        writer.add_scalar('alpha/train', model.alpha, i)

    
    torch.save(model.state_dict(), f'./saved_models/cora_encoder_{run_name}.pth')