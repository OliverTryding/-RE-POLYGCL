import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from models.PolyGCL_model import PolyGCL
from models.LogisticRegression import LogisticRegression
import torch.nn as nn
from utils2 import get_masks, EarlyStopping
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime

from typing import Union, Tuple


def evaluate_linear_classifier(model: Union[str, torch.nn.Module], dataset, device: str) -> Tuple[float, float, float, float]:

    train_nodes, val_nodes, test_nodes = get_masks(dataset.x.shape[0], train_split=0.6, val_split=0.2)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(*model(dataset[0].x, dataset[0].edge_index)).detach()

    # define simple linear classifier
    logreg = LogisticRegression(in_size=embeddings.shape[-1], n_classes=dataset.num_classes)
    logreg = logreg.to(device)

    # Train loop
    optimizer = torch.optim.Adam(logreg.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=100, mode='min')

    train_embeddings = embeddings[train_nodes]
    train_labels = dataset[0].y[train_nodes]

    val_embeddings = embeddings[val_nodes]
    val_labels = dataset[0].y[val_nodes]

    test_embeddings = embeddings[test_nodes]
    test_labels = dataset[0].y[test_nodes]

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

    print(f'LR Loss: {loss.item():.4f}, val loss: {val_loss.item(): .4f}, train acc: {train_acc: .2%}, val acc: {val_acc: .2%}')

    best_model = early_stopping.best_model
    test_logits = best_model(test_embeddings)
    test_loss = loss_fn(test_logits, test_labels)

    test_pred = test_logits.argmax(dim=-1)      
    test_acc = (test_pred == test_labels).to(torch.float32).mean()

    print(f'test acc: {test_acc.item(): .2%}, test loss: {test_loss.item(): .4f}')

    return loss.item(), test_loss.item(), train_acc.item(), test_acc.item()
