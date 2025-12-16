from models.base_network import BaseNetwork
from torch import nn
import torch
import torch.nn.functional as F

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
            activation_module = nn.LeakyReLU(inplace=True)
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

        # collect linear layer shapes and bias flags in order to support
        # vectorized forward that consumes flattened weight batches.
        self._linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        self._param_info: list[tuple[int, int, bool]] = []  # (out, in, has_bias)
        for lin in self._linear_layers:
            self._param_info.append((lin.out_features, lin.in_features, lin.bias is not None))

    def forward_with_flat_weights(self, flat_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Vectorized forward over a batch of flat weight vectors.

        Args:
            flat_weights: Tensor shaped (K, P) where P == self.num_parameters()
            x: Tensor shaped (N, input_dim)

        Returns:
            logits: Tensor shaped (K, N, output_dim)
        """
        if flat_weights.dim() != 2:
            raise ValueError("flat_weights must be shape (K, P)")
        K, P = flat_weights.shape
        expected = self.num_parameters()
        if P != expected:
            raise ValueError(f"Expected flat vector length {expected}, got {P}")

        device = x.device
        flat = flat_weights.to(device)

        # we'll walk through the parameter layout in the same order as model.parameters()
        current_idx = 0

        activations: torch.Tensor | None = None
        # iterate over layers and apply weight/bias extracted from flat
        for i, (out_dim, in_dim, has_bias) in enumerate(self._param_info):
            w_len = out_dim * in_dim
            w_flat = flat[:, current_idx: current_idx + w_len]
            w = w_flat.view(K, out_dim, in_dim)
            current_idx += w_len

            b = None
            if has_bias:
                b_flat = flat[:, current_idx: current_idx + out_dim]
                b = b_flat.view(K, out_dim)
                current_idx += out_dim

            if activations is None:
                # first layer: x is (N, in_dim)
                # compute einsum 'ni,koi->kno' -> (K, N, out_dim)
                out = torch.einsum('ni,koi->kno', x, w)
            else:
                # activations is (K, N, in_dim)
                # compute einsum 'kni,koi->kno'
                out = torch.einsum('kni,koi->kno', activations, w)

            if b is not None:
                out = out + b.unsqueeze(1)

            # apply activation (MLP uses same activation module for all hidden and output layers)
            # activation choice was stored by inspection of the net in constructor
            # use functional variants for batched tensors
            act_name = type(self._linear_layers[0 + i]).__name__  # placeholder: we use config below
            # determine activation type from the class of the activation module instance
            # fallback to leaky_relu with slope 0.01
            # We know the network uses either ReLU or LeakyReLU from constructor
            # Infer by checking instance type of first activation module in the net
            # Find an activation module in net (not linear)
            # (We do this once per forward; it's cheap) 
            activation_module = None
            for m in self.net:
                if not isinstance(m, nn.Linear):
                    activation_module = m
                    break

            if activation_module is None or isinstance(activation_module, nn.ReLU):
                out = F.relu(out)
            elif isinstance(activation_module, nn.LeakyReLU):
                # use default negative_slope=0.01
                out = F.leaky_relu(out, negative_slope=activation_module.negative_slope)
            else:
                # fallback
                out = F.leaky_relu(out, negative_slope=0.01)

            activations = out

        # activations now has shape (K, N, output_dim)
        return activations


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, output_dim).
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.net(x)
