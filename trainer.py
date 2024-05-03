from argparse import Namespace

import numpy as np
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures

from arguments import get_args
from lr_evaluation import evaluate_linear_classifier
from data_factory import get_dataset
from loss import contrastive_loss
from model_factory import get_model
from models.PolyGCL_model import PolyGCL
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
import time


def train(model, optim, data):
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


def main(args: Namespace) -> None:
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    device = args.device

    # dataset = WikipediaNetwork(root='data/chameleon', name='chameleon', transform=NormalizeFeatures())
    dataset = get_dataset(args)

    args.in_dim = dataset.x.shape[-1]

    model = get_model(args)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    data = dataset[0].to(device)

    # TODO add early stopper
    # TODO add validation and test mask -> compute loss only on train set
    # TODO add evaluation step

    for i in range(5000):
        loss = train(model, optimizer, data)

        print(model.encoder.convolution.gammas_L, "low gammas")
        print(model.encoder.convolution.gammas_H, "high gammas")

        writer.add_scalar('Loss/train', loss, i)
        # writer.add_scalar('beta/train', model.beta, i)
        writer.add_scalar('beta/train', 1 - torch.sigmoid(model.alpha), i)
        writer.add_scalar('alpha/train', torch.sigmoid(model.alpha), i)

        if i % 200 == 0:
            # Train a linear classifier on the current embeddings
            # This has no impact on the embedding training, as labels should not be known.
            log_reg_loss, log_reg_val_loss, log_reg_train_acc, log_reg_val_acc = \
                evaluate_linear_classifier(model, dataset, device=device)
            writer.add_scalar('LR_loss/train', log_reg_loss, i)
            writer.add_scalar('LR_loss/val', log_reg_val_loss, i)
            writer.add_scalar('LR_acc/train', log_reg_train_acc, i)
            writer.add_scalar('LR_acc/val', log_reg_val_acc, i)

    torch.save(model.state_dict(), f'./saved_models/cora_encoder_{run_name}.pth')


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = get_args()
    fix_seeds(args.seed)
    start = time.time()
    main(args)
    print(f"Execution time: {time.time() - start}")

