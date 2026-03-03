from unicodedata import name

from torch import nn
import torch
import timm 

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def init_weights(self, method: str = "kaiming_normal"):
        raise NotImplementedError("Weight initialization not implemented for BaseNetwork. Implement in subclass if needed.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method not implemented for BaseNetwork. Implement in subclass.")


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
        with torch.no_grad():
            return self.net(x)

    def init_weights(self):
        device = next(self.parameters()).device
        # Sample directly on the `device` (defaults to model's parameter device) to avoid copies.
        flat_weights = torch.randn(self.num_parameters(), device=device)
        flat_weights /= flat_weights.norm()

        if flat_weights.numel() != self.num_parameters():
            raise ValueError(f"Expected flat_weights of size {self.num_parameters()}, but got {flat_weights.numel()}")
        current_index = 0
        # Use in-place copy under no_grad to avoid re-allocations and keep params on their device
        with torch.no_grad():
            for param in self.parameters():
                param_length = param.numel()
                param.data.copy_(flat_weights[current_index:current_index + param_length].view_as(param))
                current_index += param_length
        

class MobileViT(BaseNetwork):
    def __init__(self, model_name: str = 'mobilevit_xxs', num_classes: int = 2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)
    

    def init_weights(self, method: str = "original"):
        """Helper function to initialize differnet layers in a model"""
        if method == "original":
            # initialize conv layers with Kaiming normal, linear layers with normal, and norm layers with constant values.
            conv_init_type = "kaiming_normal"
            conv_std = 0.01
            linear_init_type = "normal"
            linear_std = 0.01
            for module in self.modules():
                if isinstance(module, (nn.Conv2d)):
                    self.initialize_conv_layer(module=module, init_method=conv_init_type, std_val=conv_std)
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    self.initialize_norm_layers(module=module)
                elif isinstance(module, (nn.Linear)):
                    self.initialize_fc_layer(module=module, init_method=linear_init_type, std_val=linear_std)

        elif method == "unit_sphere":
            # For each weight tensor, sample from a standard normal distribution and then normalize to have unit norm.
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    self.initialize_unit_sphere(module)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
    
    def replace_normalization_with_identity(self, module: nn.Module = None):
        """In-place: replace BN/LN/GN (and BatchNormAct2d wrappers if present) with Identity."""
        if module is None:
            module = self.model  # start from the top-level model if no module is provided
        for name, child in list(module.named_children()):  # list() avoids mutation during iteration
            # Replace plain norm layers
            if isinstance(child, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                setattr(module, name, nn.Identity())
            else:
                self.replace_normalization_with_identity(child)
        
                    
        
    def initialize_unit_sphere(self, module):
        device = next(module.parameters()).device
        flat_weights = torch.randn(module.weight.numel(), device=device)
        flat_weights /= flat_weights.norm()
        with torch.no_grad():
            module.weight.data.copy_(flat_weights.view_as(module.weight))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def initialize_conv_layer(self, module: nn.Module, init_method: str, std_val: float):
        if init_method == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif init_method == "normal":
            nn.init.normal_(module.weight, std=std_val)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def initialize_norm_layers(self, module: nn.Module):
        if isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def initialize_fc_layer(self, module: nn.Module, init_method: str, std_val: float):
        if init_method == "normal":
            nn.init.normal_(module.weight, std=std_val)
        else:
            raise ValueError(f"Unsupported linear init method: {init_method}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
