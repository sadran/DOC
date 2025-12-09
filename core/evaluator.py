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
        self.device = torch.device(device)
    
    @torch.no_grad()
    def compute_error(
        self, model: torch.nn.Module, 
        loader: DataLoader) -> float:
        
        model.eval()
        model.to(self.device)

        total = 0
        incorrect = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            total += y.numel()
            incorrect += (preds != y).sum().item()

        return incorrect / total if total else 0.0
