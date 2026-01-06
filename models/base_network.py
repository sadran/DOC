from torch import nn
import torch

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def sample_unit_sphere_weights(self, device):
        # Sample directly on the `device` (defaults to model's parameter device) to avoid copies.
        flat_weights = torch.randn(self.num_parameters(), device=device)
        flat_weights /= flat_weights.norm()
        return flat_weights

    def set_flatten_weights(self, flat_weights):
        if flat_weights.numel() != self.num_parameters():
            raise ValueError(f"Expected flat_weights of size {self.num_parameters()}, but got {flat_weights.numel()}")
        current_index = 0
        # Use in-place copy under no_grad to avoid re-allocations and keep params on their device
        with torch.no_grad():
            for param in self.parameters():
                param_length = param.numel()
                param.data.copy_(flat_weights[current_index:current_index + param_length].view_as(param))
                current_index += param_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...