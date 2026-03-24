from unicodedata import name

from torch import nn
import torch


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def init_weights(self):
        raise NotImplementedError("Weight initialization not implemented for BaseNetwork. Implement in subclass if needed.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method not implemented for BaseNetwork. Implement in subclass.")
    