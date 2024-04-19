from turtle import forward
import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size, n_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Assumes that the loss function already implements the Sigmoid act
        self.linear = torch.nn.Linear(in_size, n_classes)

    
    def forward(self, x):
        return self.linear(x)