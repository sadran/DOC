from torch import nn
import torch
import timm 

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


class MLP(BaseNetwork):
    """
    Fully-connected MLP with configurable hidden layers, ReLU activations,
    and optional bias in each linear layer.

    Intended for:
      - Synthetic Gaussian experiments (10D input)
      - MNIST 1 vs 2 (784D input)

    Example:
        mlp = MLP(
            input_dim=10,
            hidden_layers=[10],
            output_dim=2,
            bias=False,
        )
    """

    def __init__(self,
                 input_dim: int,
                 hidden_layers: list[int],
                 output_dim: int,
                 activation: str = "leaky_relu",
                 bias: bool = False):
        
        super().__init__()

        if activation.lower() == "relu":
            activation_module = nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            activation_module = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        layers: list[nn.Module] = []
        prev_dim = input_dim
        # hidden layers
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h, bias=bias))
            layers.append(activation_module)
            prev_dim = h

        # output layer WITH ReLU
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        layers.append(activation_module)
        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, output_dim).
        """
        with torch.no_grad():
            return self.net(x)


class MobileViT(BaseNetwork):
    def __init__(self, model_name: str = 'mobilevit_xs', num_classes: int = 2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)
    
