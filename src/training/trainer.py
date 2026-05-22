"""
Trainer: full training loop for multi-label CXR classification.

Features:
  - Mixed-precision (AMP) via torch.cuda.amp
  - CosineAnnealingWarmRestarts LR schedule
  - Early stopping on val macro-AUROC
  - Saves last_<model>.pt every epoch + best_<model>.pt on improvement
  - Auto-resumes from last_<model>.pt if it exists, or downloads from WandB artifacts
  - Uploads last_<model>.pt to WandB artifacts after each epoch for cross-session persistence
  - WandB logging (batch loss/lr + epoch metrics)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import find_resume_checkpoint, load_checkpoint, save_checkpoint
from .wandb_logger import WandBLogger


class Trainer:
    """Manages one complete training run for a CXRClassifier.

    Args:
        model:          CXRClassifier instance (already on device).
        optimizer:      Configured AdamW with layerwise param groups.
        scheduler:      CosineAnnealingWarmRestarts instance.
        criterion:      BCEWithLogitsLoss (with pos_weight on device).
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        device:         'cuda' or 'cpu'.
        checkpoint_dir: Directory for last/best .pt files.
        model_name:     Stem used in checkpoint filenames.
        max_epochs:     Total epochs to run.
        patience:       Early-stopping patience (epochs without improvement).
        mixed_precision: Enable AMP.
        log_interval:   Log batch metrics every N steps.
        wandb_logger:   WandBLogger instance (may be disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        checkpoint_dir: str,
        model_name: str,
        max_epochs: int = 30,
        patience: int = 5,
        mixed_precision: bool = True,
        log_interval: int = 50,
        wandb_logger: Optional[WandBLogger] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.wandb_logger = wandb_logger or WandBLogger(enabled=False, project="")

        self.scaler = GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision
        self._device_type = device.split(":")[0]  # "cuda" or "cpu"

        self.start_epoch: int = 0
        self.best_auroc: float = 0.0
        self.epochs_no_improve: int = 0
        self.global_step: int = 0

    # ── Public entry-point ────────────────────────────────────────────────────

    def fit(self) -> Dict[str, float]:
        """Run the full training loop. Returns the best epoch metrics dict."""
        self._try_resume()
        self.wandb_logger.watch(self.model, log_freq=self.log_interval)

        best_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch, self.max_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch()
            elapsed = time.time() - t0

            val_auroc = val_metrics["val_auroc_macro"]
            is_best = val_auroc > self.best_auroc

            if is_best:
                self.best_auroc = val_auroc
                self.epochs_no_improve = 0
                best_metrics = {**train_metrics, **val_metrics}
            else:
                self.epochs_no_improve += 1

            all_metrics = {**train_metrics, **val_metrics, "epoch_time_s": elapsed}
            self.wandb_logger.log_epoch(epoch, all_metrics)

            save_checkpoint(
                self.checkpoint_dir,
                self.model_name,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                all_metrics,
                is_best,
                self.scaler,
            )

            last_ckpt = str(Path(self.checkpoint_dir) / f"last_{self.model_name}.pt")
            self.wandb_logger.upload_checkpoint(last_ckpt, f"last_{self.model_name}")

            self._print_epoch(epoch, train_metrics, val_metrics, elapsed, is_best)

            if self.epochs_no_improve >= self.patience:
                print(
                    f"\n[trainer] Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)."
                )
                break

        self.wandb_logger.finish()
        return best_metrics

    # ── Private helpers ───────────────────────────────────────────────────────

    def _try_resume(self) -> None:
        resume_path = find_resume_checkpoint(self.checkpoint_dir, self.model_name)

        if resume_path is None:
            print("[trainer] No local checkpoint found — checking WandB artifacts...")
            resume_path = self.wandb_logger.download_checkpoint(
                f"last_{self.model_name}", self.checkpoint_dir
            )
            if resume_path:
                print(f"[trainer] Downloaded checkpoint from WandB: {resume_path}")

        if resume_path is None:
            return
        print(f"[trainer] Resuming from {resume_path}")
        ckpt = load_checkpoint(
            resume_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
        )
        self.start_epoch = ckpt["epoch"] + 1
        self.best_auroc = ckpt["metrics"].get("val_auroc_macro", 0.0)
        self.global_step = self.start_epoch * len(self.train_loader)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", unit="batch", leave=False)
        for step, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(self._device_type, enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(epoch + step / len(self.train_loader))

            total_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if self.global_step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.wandb_logger.log_batch(
                    self.global_step,
                    {"loss": loss.item(), "lr": current_lr},
                )

        return {"train_loss": total_loss / len(self.train_loader)}

    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with autocast(self._device_type, enabled=self.mixed_precision):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                total_loss += loss.item()
                all_logits.append(logits.cpu().float().numpy())
                all_labels.append(labels.cpu().float().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        probs_np = 1 / (1 + np.exp(-logits_np))  # sigmoid

        # Per-class AUROC; skip classes with no positive samples
        per_class_auroc = []
        for c in range(labels_np.shape[1]):
            if labels_np[:, c].sum() > 0:
                per_class_auroc.append(roc_auc_score(labels_np[:, c], probs_np[:, c]))

        macro_auroc = float(np.mean(per_class_auroc)) if per_class_auroc else 0.0

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_auroc_macro": macro_auroc,
        }

    @staticmethod
    def _print_epoch(
        epoch: int,
        train: Dict[str, float],
        val: Dict[str, float],
        elapsed: float,
        is_best: bool,
    ) -> None:
        marker = " ★" if is_best else ""
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train['train_loss']:.4f} | "
            f"val_loss {val['val_loss']:.4f} | "
            f"val_auroc {val['val_auroc_macro']:.4f} | "
            f"{elapsed:.0f}s{marker}"
        )
