from torch import nn
import torch

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def sample_unit_sphere_weights(self):
        flat_weights = torch.randn(self.num_parameters())
        flat_weights /= flat_weights.norm()
        return flat_weights

    def set_flatten_weights(self, flat_weights):
        current_index = 0

        if flat_weights.numel() != self.num_parameters():
            raise ValueError(f"Expected flat_weights of size {self.num_parameters()}, but got {flat_weights.numel()}")
        
        for param in self.parameters():
            param_length = param.numel()
            param.data = flat_weights[current_index:current_index + param_length].view_as(param).data
            current_index += param_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...