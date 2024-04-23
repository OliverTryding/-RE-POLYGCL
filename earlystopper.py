from typing import Literal
import copy
import datetime
import torch

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
            path = f'./saved_models/model_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pth'

        torch.save(self.best_model.state_dict(), path)