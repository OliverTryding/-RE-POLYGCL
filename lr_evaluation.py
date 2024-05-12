import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from models.PolyGCL_model import PolyGCL
from models.LogisticRegression import LogisticRegression
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from utils2 import get_masks, EarlyStopping, random_splits
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime

import seaborn as sns

from typing import Union, Tuple


def evaluate_linear_classifier(model: Union[str, torch.nn.Module], dataset, args, verbose=True, seed: int = 42) -> Tuple[float, float, float, float]:
    device = args.device
    label = dataset[0].y
    n_classes = dataset.num_classes
    
    train_rate = 0.6
    val_rate = 0.2


    if args.dataname in ['roman_empire', 'amazon_ratings', "minesweeper", "tolokers", "questions"]:
            data = dataset[0]
            train_nodes, val_nodes, test_nodes = data.train_mask[:, i].to(args.device), data.val_mask[:, i].to(args.device), data.test_mask[:, i].to(args.device)
            n_classes = n_classes if args.dataname in ['roman_empire', 'amazon_ratings'] else 1
            if args.dataname in ["minesweeper", "tolokers", "questions"]:
                label = label.to(torch.float)
    else:
        percls_trn = int(round(train_rate * len(label) / n_classes))
        val_lb = int(round(val_rate * len(label)))

        train_nodes, val_nodes, test_nodes = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(*model(dataset[0].x, dataset[0].edge_index)).detach()

    # define simple linear classifier
    logreg = LogisticRegression(in_size=embeddings.shape[-1], n_classes=dataset.num_classes)
    logreg = logreg.to(device)

    # Train loop
    optimizer = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    loss_fn = torch.nn.BCEWithLogitsLoss() if args.dataname in ["minesweeper", "tolokers", "questions"] else torch.nn.CrossEntropyLoss()


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
        if args.dataname in ["minesweeper", "tolokers", "questions"]:
            logits = logits.squeeze(-1)
        loss = loss_fn(logits, train_labels)

        loss.backward()
        optimizer.step()

        # validation step
        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embeddings)
            val_loss = loss_fn(val_logits, val_labels)

            if early_stopping(val_loss.item(), logreg):
                if verbose:
                    print('Early stopped!')
                break

            # stats
            train_pred = logits.argmax(dim=-1)      
            train_acc = (train_pred == train_labels).to(torch.float32).mean()

            val_pred = val_logits.argmax(dim=-1)      
            val_acc = (val_pred == val_labels).to(torch.float32).mean()

            if args.dataname in ["minesweeper", "tolokers", "questions"]:
                val_acc = roc_auc_score(y_true=val_labels.cpu().numpy(), y_score=val_logits.squeeze(-1).cpu().numpy())


    if verbose:
        print(f'LR Loss: {loss.item():.4f}, val loss: {val_loss.item(): .4f}, train acc: {train_acc: .2%}, val acc: {val_acc: .2%}')

    best_model = early_stopping.best_model
    test_logits = best_model(test_embeddings)
    test_loss = loss_fn(test_logits, test_labels)

    test_pred = test_logits.argmax(dim=-1)      
    test_acc = (test_pred == test_labels).to(torch.float32).mean()
    if args.dataname in ["minesweeper", "tolokers", "questions"]:
        test_acc = roc_auc_score(y_true=test_labels.cpu().numpy(), y_score=test_logits.squeeze(-1).cpu().numpy())


    if verbose:
        print(f'test acc: {test_acc.item(): .2%}, test loss: {test_loss.item(): .4f}')

    return loss.item(), test_loss.item(), train_acc.item(), test_acc.item()
