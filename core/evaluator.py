# core/evaluator.py

from __future__ import annotations
import torch
from torch import nn

def _is_trivial_classifier(preds) -> bool:
    # Check if all predictions are the same (i.e., the model is trivial)
    return torch.all(preds == preds[0])

class Evaluator:
    """
    Provides evaluation utilities for computing error.
    """

    def __init__(self, device: str = "cpu"):
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available.")
        self.device = torch.device(device)
    
    @torch.no_grad()
    def compute_error(self, model: torch.nn.Module, loader: object):
        """Compute the error of the model on the given data loader."""
        model.eval()
        total = 0
        incorrect = 0
        for data, target in loader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            #with torch.amp.autocast(device_type=self.device.type):
            logits = model(data)
            preds = logits.argmax(dim=1)
            total += target.size(0)
            incorrect += (preds != target).sum().item()
            error = incorrect / total if total else 0.0
        return error
    
    def compute_error_and_trivial_flag(self, model: torch.nn.Module, dataset: object):
        """Convenience method to compute error and trivial classifier flag directly on a dataset with .x and .y tensors, without needing to create a DataLoader.
        """
        x = dataset.x.to(self.device, non_blocking=True)
        y = dataset.y.to(self.device, non_blocking=True)
        #with torch.amp.autocast(device_type=self.device.type):
        logits = model(x)
        preds = logits.argmax(dim=1)
        total = y.numel()
        incorrect = (preds != y).sum().item()
        error = incorrect / total if total else 0.0
        trivial_classifier_flag = torch.all(preds == preds[0])
        return error, trivial_classifier_flag
    
    @torch.no_grad()
    def compute_error_on_dataset(self, model: torch.nn.Module, dataset: object):
        """Convenience method to compute error directly on a dataset with .x and .y tensors, without needing to create a DataLoader.
        for the DOC experiments, it's way faster to compute error directly on the dataset tensor rather than loading data from dataloader.
        """
        x = dataset.x.to(self.device, non_blocking=True)
        y = dataset.y.to(self.device, non_blocking=True)
        #with torch.amp.autocast(device_type=self.device.type):
        logits = model(x)
        preds = logits.argmax(dim=1)
        total = y.numel()
        incorrect = (preds != y).sum().item()
        error = incorrect / total if total else 0.0
        return error
