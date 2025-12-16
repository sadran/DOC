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

    def sample_unit_sphere_weights_batch(self, k: int):
        """Return a batch of k flat weight vectors sampled uniformly on the unit sphere.

        Returns a tensor of shape (k, num_parameters), each row normalized independently.
        """
        flat = torch.randn(k, self.num_parameters())
        norms = flat.norm(dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        flat = flat / norms
        return flat

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