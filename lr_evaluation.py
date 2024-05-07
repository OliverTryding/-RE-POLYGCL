import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from models.PolyGCL_model import PolyGCL
from models.LogisticRegression import LogisticRegression
import torch.nn as nn
from utils2 import get_masks, EarlyStopping, random_splits
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime

import seaborn as sns

from typing import Union, Tuple


def evaluate_post_training(args, label, embeds, n_classes):
    print("=== Evaluation ===")
    ''' Linear Evaluation '''
    results = []
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(label) / n_classes))
    val_lb = int(round(val_rate * len(label)))
    for i in range(10):
        seed = SEEDS[i]
        train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

        train_mask = torch.BoolTensor(train_mask).to(args.device)
        val_mask = torch.BoolTensor(val_mask).to(args.device)
        test_mask = torch.BoolTensor(test_mask).to(args.device)

        train_embs = embeds[train_mask]
        val_embs = embeds[val_mask]
        test_embs = embeds[test_mask]

        label = label.to(args.device)

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogisticRegression(in_size=args.hid_dim, n_classes=n_classes)
        opt = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = torch.argmax(logits, dim=1)
            train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with torch.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        print(i, 'Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        results.append(eval_acc.cpu().data)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')


def evaluate_linear_classifier(model: Union[str, torch.nn.Module], dataset, device: str, verbose=True, seed: int = 42) -> Tuple[float, float, float, float]:

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
                if verbose:
                    print('Early stopped!')
                break

        # stats
        train_pred = logits.argmax(dim=-1)      
        train_acc = (train_pred == train_labels).to(torch.float32).mean()

        val_pred = val_logits.argmax(dim=-1)      
        val_acc = (val_pred == val_labels).to(torch.float32).mean()

    if verbose:
        print(f'LR Loss: {loss.item():.4f}, val loss: {val_loss.item(): .4f}, train acc: {train_acc: .2%}, val acc: {val_acc: .2%}')

    best_model = early_stopping.best_model
    test_logits = best_model(test_embeddings)
    test_loss = loss_fn(test_logits, test_labels)

    test_pred = test_logits.argmax(dim=-1)      
    test_acc = (test_pred == test_labels).to(torch.float32).mean()

    if verbose:
        print(f'test acc: {test_acc.item(): .2%}, test loss: {test_loss.item(): .4f}')

    return loss.item(), test_loss.item(), train_acc.item(), test_acc.item()