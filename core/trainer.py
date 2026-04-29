import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    error: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_error": [],
            "val_loss": [],
            "val_acc": [],
            "val_error": [],
        }

    def _move_batch_to_device(self, batch):
        inputs, targets = batch
        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def _compute_num_correct(self, outputs: torch.Tensor, targets: torch.Tensor) -> int:
        preds = outputs.argmax(dim=1)
        return (preds == targets).sum().item()

    def train_one_epoch(self, train_loader: DataLoader) -> EpochMetrics:
        self.model.train()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch in train_loader:
            inputs, targets = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_correct += self._compute_num_correct(outputs, targets)
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        epoch_error = 1.0 - epoch_acc

        return EpochMetrics(loss=epoch_loss, accuracy=epoch_acc, error=epoch_error)

    @torch.inference_mode()
    def evaluate(self, data_loader: DataLoader) -> EpochMetrics:
        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch in data_loader:
            inputs, targets = self._move_batch_to_device(batch)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_correct += self._compute_num_correct(outputs, targets)
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        epoch_error = 1.0 - epoch_acc

        return EpochMetrics(loss=epoch_loss, accuracy=epoch_acc, error=epoch_error)

    def fit_until_zero_empirical_error(
        self,
        train_loader: DataLoader,
        max_epochs: int = 1000,
        save_best_path: Optional[str] = None,
        verbose: bool = True):  
        best_state = None
        best_train_error = float("inf")
        converged = False

        for epoch in range(1, max_epochs + 1):
        #epoch = 0
        #while not converged:
            train_metrics = self.train_one_epoch(train_loader)

            self.history["train_loss"].append(train_metrics.loss)
            self.history["train_acc"].append(train_metrics.accuracy)
            self.history["train_error"].append(train_metrics.error)

            msg = (
                f"Epoch [{epoch}/{max_epochs}] "
                f"Train Loss: {train_metrics.loss:.6f}, "
                f"Train Acc: {train_metrics.accuracy:.6f}, "
                f"Train Error: {train_metrics.error:.6f}"
            )

            if train_metrics.error < best_train_error:
                best_train_error = train_metrics.error
                best_state = copy.deepcopy(self.model.state_dict())

                if save_best_path is not None:
                    torch.save(best_state, save_best_path)

            if verbose:
                print(msg)

            if train_metrics.error == 0.0:
                converged = True
                if verbose:
                    print(f"Reached zero empirical error at epoch {epoch}.")
                break

            if self.scheduler is not None:
                self.scheduler.step()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return converged
    
    def fit_n_steps_after_zero_training_error(
            self,
            train_loader: DataLoader
            
    )

    @torch.inference_mode()
    def predict(self, data_loader: DataLoader):
        self.model.eval()

        all_preds = []
        all_targets = []

        for batch in data_loader:
            inputs, targets = self._move_batch_to_device(batch)
            outputs = self.model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        return torch.cat(all_preds), torch.cat(all_targets)

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)