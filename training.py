from argparse import Namespace

import numpy as np

from arguments import get_args
from lr_evaluation import evaluate_linear_classifier
from post_eval import post_eval
from loss import contrastive_loss
from model_factory import get_model
from models.PolyGCL_model import PolyGCL
import torch
import datetime
import random
import time
from utils2 import EarlyStopping
import wandb


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

    device = args.device

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

    run_name = f'run_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.dataname}'

    wandb_run = wandb.init(project="RE-PolyGCL", entity="nihermann", name=run_name, config=args)

    data = dataset[0].to(device)

    early_stopping = EarlyStopping(patience=100, mode='min')

    for i in range(args.epochs):
        loss = train(model, optimizer, data)

        gamma_l = model.encoder.convolution.gammas_L
        gamma_h = model.encoder.convolution.gammas_H

        wandb_run.log({'gamma/low': wandb.Histogram(gamma_l.detach().cpu().numpy()), "epoch": i}, commit=False)
        wandb_run.log({'gamma/high': wandb.Histogram(gamma_h.detach().cpu().numpy()), "epoch": i}, commit=False)

        wandb_run.log({'Loss/train': loss, "epoch": i}, commit=False)

        wandb_run.log({'beta/train':  model.beta.item(), "epoch": i}, commit=False)
        wandb_run.log({'alpha/train':  model.alpha.item(), "epoch": i})

        with torch.no_grad():
            if early_stopping(loss, model):
                break

        if i % 20 == 0:
            print(f'Epoch: {i}, Loss: {loss:.4f}')

        if i % 100 == 0:
            # Train a linear classifier on the current embeddings
            # This has no impact on the embedding training, as labels should not be known.
            log_reg_loss, log_reg_test_loss, log_reg_train_acc, log_reg_test_acc = \
                evaluate_linear_classifier(model, dataset, args, verbose=True)
            wandb_run.log({'LR_loss/train':   log_reg_loss, "epoch": i}, commit=False)
            wandb_run.log({'LR_loss/test':   log_reg_test_loss, "epoch": i}, commit=False)
            wandb_run.log({'LR_acc/train':   log_reg_train_acc, "epoch": i}, commit=False)
            wandb_run.log({'LR_acc/test':   log_reg_test_acc, "epoch": i})

    model = early_stopping.best_model
    post_test_acc_mean = post_eval(model, dataset, args)

    wandb_run.log({'post_acc_mean/test':   post_test_acc_mean})


def fix_seeds(seed):
    """
    Fix seeds for reproducibility the same way as the authors did
    """
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

