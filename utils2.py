from typing import Literal
import copy
import datetime

import numpy as np
import torch


def get_masks(n_nodes, train_split=0.8, val_split=0.1):
    train_n = int(train_split * n_nodes)
    val_n = int(val_split * n_nodes)
    test_n = n_nodes - train_n - val_n

    permutation = torch.randperm(n_nodes)

    train_nodes = permutation[:train_n]
    val_nodes = permutation[train_n:train_n+val_n]
    test_nodes = permutation[train_n+val_n:]

    return train_nodes, val_nodes, test_nodes


def index_to_mask(index, size):
    # taken from authors code
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    # taken from authors code
    num_nodes = label.shape[0]
    index=[i for i in range(num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(label.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=num_nodes)
    val_mask = index_to_mask(val_idx,size=num_nodes)
    test_mask = index_to_mask(test_idx,size=num_nodes)
    return train_mask, val_mask, test_mask


class EarlyStopping:

    def __init__(self, patience=10, mode: Literal['max', 'min'] = 'min'):
        self.patience = patience

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model = None
        self.mode = mode

        self.counter = 0
        self.call_counter = 0
        self.best_score_iteration = 0

    def better(self, new, current):
        if self.mode == 'min':
            return new < current
        return new > current

    def __call__(self, score, model):
        self.call_counter += 1

        if self.better(new=score, current=self.best_score):
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.counter = 0
            self.best_score_iteration = self.call_counter
            return False
        
        self.counter += 1

        if self.counter > self.patience:
            return True
        return False
    
    def save_best_model(self, path=None):
        if path is None:
            path = f'./saved_models/model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

        torch.save(self.best_model.state_dict(), path)