import torch
import torch.nn as nn
from models.PolyGCL_model import PolyGCL
from models.LogisticRegression import LogisticRegression
from loss import contrastive_loss
from trainer import train

from torch_geometric.datasets import Planetoid, WikipediaNetwork
from utils import get_masks, EarlyStopping

from typing import Union, Tuple
import datetime
import sys

from torch.utils.tensorboard import SummaryWriter

def evaluate_linear_classifier(embeddings, dataset, train_nodes: torch.Tensor, val_nodes: torch.Tensor, test_nodes: torch.Tensor, device="cuda") -> Tuple[float, float]:
    
    dataset.to(device)

    # define simple linear classifier
    logreg = LogisticRegression(in_size=embeddings.shape[-1], n_classes=dataset.num_classes)
    logreg = logreg.to(device)

    # Train loop
    optimizer = torch.optim.Adam(logreg.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=50, mode='min')

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
                break

    best_model = early_stopping.best_model
    test_logits = best_model(test_embeddings)
    test_loss = loss_fn(test_logits, test_labels)

    test_pred = test_logits.argmax(dim=-1)      
    test_acc = (test_pred == test_labels).to(torch.float32).mean()

    return test_loss.item(), test_acc.item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042, 2424918363]

    print("Training")

    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0].to(device)
    train_nodes, val_nodes, test_nodes = get_masks(dataset.x.shape[0],train_split=0.6, val_split=0.2)

    model = PolyGCL(in_size=dataset[0].x.shape[-1], hidden_size=512, out_size=512, K=10, dropout_p=0.3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        early_stopping = EarlyStopping(patience=50, mode='min')

        run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{seed}'
        writer = SummaryWriter(log_dir=f'runs/{run_name}')

        for i in range(1000):
            loss = train(model, optimizer, data)

            writer.add_scalar('Loss/train', loss, i)
            # writer.add_scalar('beta/train', model.beta, i)
            writer.add_scalar('beta/train', 1-torch.nn.functional.sigmoid(model.alpha), i)
            writer.add_scalar('alpha/train', torch.nn.functional.sigmoid(model.alpha), i)

            with torch.no_grad():
                if early_stopping(loss, model):
                    break

            if i % 20 == 0:
                # print aligned
                print(f'Epoch: {i}, Loss: {loss:.4f}')

        model = early_stopping.best_model
        models.append(model)

    print("Evaluation")

    accs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embedding(*model(data.x, data.edge_index)).detach()

        # Train a linear classifier on the current embeddings
        # This has no impact on the embedding training, as labels should not be known. 
        log_reg_test_loss, log_reg_test_acc = evaluate_linear_classifier(embeddings, dataset, train_nodes, val_nodes, test_nodes, device=device)
        print(f'Linear classifier test loss: {log_reg_test_loss:.4f}, test accuracy: {log_reg_test_acc:.4f}')
        writer.add_scalar('LR_loss/test',log_reg_test_loss, i)
        writer.add_scalar('LR_acc/test', log_reg_test_acc, i)
        accs.append(log_reg_test_acc)

    # Print the average accuracy and the standard deviation
    print(f'Average accuracy: {sum(accs)/len(accs):.4f} +- {torch.tensor(accs).std():.4f}')
