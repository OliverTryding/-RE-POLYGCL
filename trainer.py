from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from cora_evaluation import evaluate_linear_classifier
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





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    # dataset = WikipediaNetwork(root='data/chameleon', name='chameleon', transform=NormalizeFeatures())
    model = PolyGCL(in_size=dataset.x.shape[-1], hidden_size=512, out_size=512, K=10, dropout_p=0.3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    data = dataset[0].to(device)

    # TODO add early stopper
    # TODO add validation and test mask -> compute loss only on train set
    # TODO add evaluation step

    for i in range(1000):
        loss = train(model,optimizer,data)
        print(loss)

        print(model.encoder.convolution.gammas_L, "low gammas")
        print(model.encoder.convolution.gammas_H, "high gammas")

        writer.add_scalar('Loss/train', loss, i)
        # writer.add_scalar('beta/train', model.beta, i)
        writer.add_scalar('beta/train', 1-torch.nn.functional.sigmoid(model.alpha), i)
        writer.add_scalar('alpha/train', torch.nn.functional.sigmoid(model.alpha), i)
        
        if i % 200 == 0:
            # Train a linear classifier on the current embeddings
            # This has no impact on the embedding training, as labels should not be known. 
            log_reg_test_loss, log_reg_test_acc = evaluate_linear_classifier(model, verbose=False, use_tensorboard=False, device=device)
            writer.add_scalar('LR_loss/test',log_reg_test_loss, i)
            writer.add_scalar('LR_acc/test', log_reg_test_acc, i)

    
    torch.save(model.state_dict(), f'./saved_models/cora_encoder_{run_name}.pth')