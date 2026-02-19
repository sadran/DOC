# core/evaluator.py

from __future__ import annotations
import torch
from torch import nn


class Evaluator:
    """
    Provides evaluation utilities for:
      - true error on a fixed test set
    """

    def __init__(self, device: str = "cpu"):
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available.")
        self.device = torch.device(device)
    
    @torch.no_grad()
    def compute_error(
        self, model: torch.nn.Module, 
        loader: object) -> float:
        
        model.eval()
        """
        total = 0
        incorrect = 0
        for data, target in loader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type):
                logits = model(data)
            preds = logits.argmax(dim=1)
            total += target.size(0)
            incorrect += (preds != target).sum().item()
        return incorrect / total if total else 0.0
        """
        # For performance we assume the caller places the model on the desired device once.
        # Fast-path: if the DataLoader wraps a small dataset with tensors (`.x`, `.y`), evaluate in a single pass.
        dataset = getattr(loader, "dataset", None)
        #if dataset is not None and hasattr(dataset, "x") and hasattr(dataset, "y"):
        x = dataset.x.to(self.device, non_blocking=True)
        y = dataset.y.to(self.device, non_blocking=True)
        with torch.amp.autocast(device_type=self.device.type):
            logits = model(x)
        preds = logits.argmax(dim=1)
        total = y.numel()
        incorrect = (preds != y).sum().item()
        return incorrect / total if total else 0.0

