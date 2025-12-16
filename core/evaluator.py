# core/evaluator.py

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torch import nn


class Evaluator:
    """
    Provides evaluation utilities for:
      - true error on a fixed test set
    """

    def __init__(self, device: str = "cpu"):
        # prefer cuda when requested but fall back to cpu if not available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
    
    @torch.no_grad()
    def compute_error(
        self, model: torch.nn.Module, 
        loader: DataLoader) -> float:
        
        model.eval()
        model.to(self.device)

        total = 0
        incorrect = 0

        use_amp = self.device.type == 'cuda'
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
            else:
                logits = model(x)
            preds = logits.argmax(dim=1)

            total += y.numel()
            incorrect += (preds != y).sum().item()

        return incorrect / total if total else 0.0

    @torch.no_grad()
    def compute_error_vectorized(self, model: torch.nn.Module, flat_weights_batch: torch.Tensor, loader: DataLoader) -> torch.Tensor:
        """Compute vectorized true errors for a batch of flattened weight vectors.

        Returns a tensor of shape (K,) with error rate for each weight sample.
        """
        device = self.device
        model.to(device)

        K = flat_weights_batch.shape[0]
        incorrect = torch.zeros(K, device=device, dtype=torch.long)
        total = 0

        use_amp = device.type == 'cuda'

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model.forward_with_flat_weights(flat_weights_batch.to(device), x)
            else:
                logits = model.forward_with_flat_weights(flat_weights_batch.to(device), x)

            preds = logits.argmax(dim=2)  # (K, N_batch)
            neq = (preds != y.unsqueeze(0)).sum(dim=1)
            incorrect += neq.to(device)
            total += y.numel()

        if total == 0:
            return torch.zeros(K, device=device)
        return incorrect.float() / float(total)
