import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from models.PolyGCL_model import PolyGCL
from models.LogisticRegression import LogisticRegression
import torch.nn as nn
from utils import get_masks, EarlyStopping
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime

from typing import Union, Tuple


def evaluate_linear_classifier(model: Union[str, torch.nn.Module], verbose=True, use_tensorboard=True, device="cuda") -> Tuple[float, float, float, float]:
    # dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    dataset = WikipediaNetwork(root='data/chameleon', name='chameleon', transform=NormalizeFeatures())

    train_nodes, val_nodes, test_nodes = get_masks(dataset.x.shape[0])
    print(f'Train size: {dataset.x[train_nodes].shape}')
    print(f'Val size: {dataset.x[val_nodes].shape}')
    print(f'Test size: {dataset.x[test_nodes].shape}')

    # load encoder model
    if isinstance(model, str):
        model = PolyGCL(in_size=dataset.x.shape[-1], hidden_size=128, out_size=128, K=10, dropout_p=0.4)
        try:
            path = sys.argv[1]
            model.load_state_dict(torch.load(path))
        except:
            print('Error: Provide the embedding model path as first argument')
            exit()
                
    dataset = dataset.to(device)
    
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(*model(dataset[0].x, dataset[0].edge_index)).detach()

    # define simple linear classifier
    logreg = LogisticRegression(in_size=embeddings.shape[-1], n_classes=dataset.num_classes)
    logreg = logreg.to(device)

    # Train loop
    optimizer = torch.optim.Adam(logreg.parameters(), lr=5e-2)
    loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=40, mode='min')

    train_embeddings = embeddings[train_nodes]
    train_labels = dataset[0].y[train_nodes]

    val_embeddings = embeddings[val_nodes]
    val_labels = dataset[0].y[val_nodes]

    test_embeddings = embeddings[test_nodes]
    test_labels = dataset[0].y[test_nodes]

    if use_tensorboard:
        run_name = f'{dataset.name}_eval_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
        writer = SummaryWriter(log_dir=f'runs/{run_name}')

    epochs = 1000

    for e in range(epochs):
        # training step
        logreg.train()
        optimizer.zero_grad()

        logits = logreg(train_embeddings)

        loss = loss_fn(logits, train_labels)
        

        loss.backward()
        optimizer.step()

        # validation step
        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embeddings)
            val_loss = loss_fn(val_logits, val_labels)
            

            if early_stopping(val_loss.item(), logreg):
                print('Early stopped!')
                break

        # stats
        train_pred = logits.argmax(dim=-1)      
        train_acc = (train_pred == train_labels).to(torch.float32).mean()

        val_pred = val_logits.argmax(dim=-1)      
        val_acc = (val_pred == val_labels).to(torch.float32).mean()

        if verbose:
            print(f'Train loss: {loss.item()}, val loss: {val_loss.item()}, train acc: {train_acc}, val acc: {val_acc}' )
        if use_tensorboard:
            writer.add_scalar('loss/train', loss, e)
            writer.add_scalar('loss/val', val_loss, e)
            writer.add_scalar('acc/train', train_acc, e)
            writer.add_scalar('acc/val', val_acc, e)
    
    print(f'LR Loss: {loss.item():.4f}, val loss: {val_loss.item(): .4f}, train acc: {train_acc: .2%}, val acc: {val_acc: .2%}')
    return loss.item(), val_loss.item(), train_acc.item(), val_acc.item()


if __name__ == '__main__':
    evaluate_linear_classifier()