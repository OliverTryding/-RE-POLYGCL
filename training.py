from argparse import Namespace

import numpy as np
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures

from arguments import get_args
from lr_evaluation import evaluate_linear_classifier
from post_eval import post_eval
from loss import contrastive_loss
from model_factory import get_model
from models.PolyGCL_model import PolyGCL
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
import time
# import nvidia_smi
from utils2 import EarlyStopping


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
    from data_factory import get_dataset

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    # if "cuda" in args.device:
    #     nvidia_smi.nvmlInit()
    #     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    #     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    device = args.device

    # dataset = WikipediaNetwork(root='data/chameleon', name='chameleon', transform=NormalizeFeatures())
    dataset = get_dataset(args).to(device)

    args.in_dim = dataset.x.shape[-1]

    model = get_model(args)
    model = model.to(device)

    optimizer = torch.optim.Adam([{'params': model.encoder.up.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.encoder.input_encoder.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.discriminator.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.encoder.convolution.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
                                  ])
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.dataname}'

    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    data = dataset[0].to(device)

    # TODO add early stopper
    # TODO add validation and test mask -> compute loss only on train set
    # TODO add evaluation step
    early_stopping = EarlyStopping(patience=50, mode='min')

    for i in range(args.epochs):
        loss = train(model, optimizer, data)

        gamma_l = model.encoder.convolution.gammas_L
        gamma_h = model.encoder.convolution.gammas_H
        # print(model.encoder.convolution.gammas_H, "high gammas")
        writer.add_histogram('gamma/low', gamma_l, i)
        writer.add_histogram('gamma/high', gamma_h, i)

        writer.add_scalar('Loss/train', loss, i)
        # writer.add_scalar('beta/train', model.beta, i)
        writer.add_scalar('beta/train', model.alpha, i)
        writer.add_scalar('alpha/train', model.alpha, i)

        # if "cuda" in args.device:
        #     writer.add_scalar('nvidia/free', info.free, i)
        #     writer.add_scalar('nvidia/used', info.used, i)
        #     writer.add_scalar('nvidia/total', info.total, i)

        with torch.no_grad():
            if early_stopping(loss, model):
                break

        if i % 20 == 0:
            # print aligned
            print(f'Epoch: {i}, Loss: {loss:.4f}')

        if i % 100 == 0:
            # Train a linear classifier on the current embeddings
            # This has no impact on the embedding training, as labels should not be known.
            log_reg_loss, log_reg_test_loss, log_reg_train_acc, log_reg_test_acc = \
                evaluate_linear_classifier(model, dataset, device=device, verbose=True)
            writer.add_scalar('LR_loss/train', log_reg_loss, i)
            writer.add_scalar('LR_loss/test', log_reg_test_loss, i)
            writer.add_scalar('LR_acc/train', log_reg_train_acc, i)
            writer.add_scalar('LR_acc/test', log_reg_test_acc, i)

    model = early_stopping.best_model
    post_eval(model, dataset, args)


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

