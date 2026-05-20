"""
Checkpoint save / load utilities.

Two files are always maintained:
  - last_<model>.pt  — saved every epoch so a crash can be resumed
  - best_<model>.pt  — saved only when val_auroc_macro improves

Resume logic: if last_<model>.pt exists, load it and continue from that epoch.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    """Save last checkpoint every epoch; overwrite best checkpoint on improvement.

    Args:
        checkpoint_dir: Directory to write .pt files into.
        model_name:     Used as filename stem (e.g. 'densenet121').
        model:          Model whose state_dict to save.
        optimizer:      Optimizer state (needed to resume training).
        scheduler:      LR scheduler state.
        epoch:          Current epoch index (0-based).
        metrics:        Dict of metric name → value logged alongside weights.
        is_best:        Whether this epoch achieved the best val AUROC so far.
        scaler:         AMP GradScaler state (None if not using mixed precision).
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }

    last_path = ckpt_dir / f"last_{model_name}.pt"
    torch.save(state, last_path)

    if is_best:
        best_path = ckpt_dir / f"best_{model_name}.pt"
        torch.save(state, best_path)
        print(f"  [ckpt] Best checkpoint saved → {best_path}  (epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        checkpoint_path: Path to a .pt file written by save_checkpoint().
        model:           Model instance to load weights into.
        optimizer:       If provided, restore optimizer state (for resuming training).
        scheduler:       If provided, restore scheduler state.
        scaler:          If provided, restore AMP scaler state.
        device:          Target device string ('cpu', 'cuda').

    Returns:
        The full checkpoint dict (contains 'epoch', 'metrics', etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    print(f"  [ckpt] Loaded checkpoint from epoch {ckpt['epoch']}  ({checkpoint_path})")
    return ckpt


def find_resume_checkpoint(checkpoint_dir: str, model_name: str) -> Optional[str]:
    """Return the path to last_<model>.pt if it exists, else None."""
    path = Path(checkpoint_dir) / f"last_{model_name}.pt"
    return str(path) if path.exists() else None
