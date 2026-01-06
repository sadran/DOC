from models.base_network import BaseNetwork
from torch import nn
import torch

class MLP(BaseNetwork):
    """
    Fully-connected MLP with configurable hidden layers, ReLU activations,
    and optional bias in each linear layer.

    Intended for:
      - Synthetic Gaussian experiments (10D input)
      - MNIST 1 vs 2 (784D input)

    Example:
        mlp = MLP(
            input_dim=784,
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
            activation_module = nn.LeakyReLU(inplace=True, negative_slope=0.1)
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
