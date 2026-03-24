from torch import nn
import torch
import timm 
from .base_network import BaseNetwork

class MobileViT(BaseNetwork):
    def __init__(self, model_config: dict):
        super().__init__()
        self.model = timm.create_model(model_config['name'], pretrained=False, num_classes=model_config['num_classes'])
        self.model_config = model_config
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

    def init_weights(self):
        """Helper function to initialize differnet layers in a model"""
        conv_init_method = self.model_config.get('conv_init_method', 'kaiming_normal')
        linear_init_method = self.model_config.get('linear_init_method', 'normal')
        norm_init_method = self.model_config.get('norm_init_method', 'ones')
        conv_std = 0.01
        linear_std = 0.01
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                self.initialize_conv_layer(module=module, init_method=conv_init_method, std_val=conv_std)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self.initialize_norm_layers(module=module, init_method=norm_init_method)
            elif isinstance(module, (nn.Linear)):
                self.initialize_fc_layer(module=module, init_method=linear_init_method, std_val=linear_std)
    
    def replace_norm_layers_with_identity(self, module: nn.Module = None):
        """In-place: replace BN/LN/GN (and BatchNormAct2d wrappers if present) with Identity."""
        if module is None:
            module = self.model  # start from the top-level model if no module is provided
        for name, child in list(module.named_children()):  # list() avoids mutation during iteration
            # Replace plain norm layers
            if isinstance(child, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                setattr(module, name, nn.Identity())
            else:
                self.replace_norm_layers_with_identity(child)
        
        
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
        elif init_method == "unit_sphere":
            self.initialize_unit_sphere(module)
        else:
            raise ValueError(f"Unsupported conv init method: {init_method}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def initialize_norm_layers(self, module: nn.Module, init_method: str):
        if init_method == "ones":
            nn.init.ones_(module.weight)
        elif init_method == "unit_sphere":
            self.initialize_unit_sphere(module)
        else:
            raise ValueError(f"Unsupported norm init method: {init_method}")
        if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def initialize_fc_layer(self, module: nn.Module, init_method: str, std_val: float):
        if init_method == "normal":
            nn.init.normal_(module.weight, std=std_val)
        elif init_method == "truncated_normal":
            nn.init.trunc_normal_(module.weight, std=std_val)
        elif init_method == "unit_sphere":
            self.initialize_unit_sphere(module)
        else:
            raise ValueError(f"Unsupported linear init method: {init_method}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)