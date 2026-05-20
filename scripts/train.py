"""
Entry-point for training DenseNet-121 or ViT-Base/16 on NIH ChestX-ray14.

Usage:
    python scripts/train.py --config configs/train.yaml --model densenet121
    python scripts/train.py --config configs/train.yaml --model vit_base_patch16_224
    python scripts/train.py --config configs/train.yaml --model densenet121 --debug
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_nih14_loaders
from src.models import build_model, get_layerwise_param_groups
from src.training import Trainer, WandBLogger, build_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CXR classifier")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--model", default=None,
                   help="Override model name from config (densenet121 | vit_base_patch16_224)")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug_mode: 20 K subset, 2 epochs")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI --debug overrides config flag
    if args.debug:
        cfg["debug_mode"] = True

    # CLI --model overrides config
    if args.model:
        cfg["model"]["name"] = args.model

    model_name: str = cfg["model"]["name"]
    debug: bool = cfg["debug_mode"]
    seed: int = cfg["training"]["seed"]

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] model={model_name}  debug={debug}  device={device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    subset_n = cfg["debug_subset_size"] if debug else None
    epochs   = cfg["debug_epochs"]      if debug else cfg["training"]["epochs"]

    train_loader, val_loader, _ = build_nih14_loaders(
        image_dir      = cfg["data"]["nih14_images"],
        labels_csv     = cfg["data"]["nih14_labels"],
        train_val_list = cfg["data"]["nih14_train_val_list"],
        test_list      = cfg["data"]["nih14_test_list"],
        batch_size     = cfg["training"]["batch_size"],
        num_workers    = cfg["training"]["num_workers"],
        img_size       = cfg["input"]["train_size"],
        debug_n        = subset_n,
        seed           = seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        name        = model_name,
        num_classes = cfg["model"]["num_classes"],
        pretrained  = cfg["model"]["pretrained"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    param_groups = get_layerwise_param_groups(
        model,
        base_lr      = cfg["training"]["lr"],
        decay_factor = cfg["training"]["layerwise_lr_decay"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["training"]["T_0"]
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    pos_weight = None
    if cfg["training"]["use_pos_weight"]:
        # Compute from training subset via the underlying NIH14Dataset
        ds = train_loader.dataset
        # Unwrap Subset if needed
        base_ds = getattr(ds, "dataset", ds)
        pos_weight = base_ds.compute_pos_weights().to(device)

    criterion = build_loss(pos_weight)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    logger = WandBLogger(
        enabled  = wandb_cfg.get("enabled", False) and not debug,
        project  = wandb_cfg.get("project", "cxr-xai-clinical"),
        entity   = wandb_cfg.get("entity", None),
        run_name = f"{model_name}_{'debug' if debug else 'full'}",
        tags     = wandb_cfg.get("tags", []),
        config   = {
            "model":      model_name,
            "epochs":     epochs,
            "batch_size": cfg["training"]["batch_size"],
            "lr":         cfg["training"]["lr"],
            "debug":      debug,
        },
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        optimizer       = optimizer,
        scheduler       = scheduler,
        criterion       = criterion,
        train_loader    = train_loader,
        val_loader      = val_loader,
        device          = device,
        checkpoint_dir  = cfg["checkpoint_dir"],
        model_name      = model_name,
        max_epochs      = epochs,
        patience        = cfg["training"]["early_stopping_patience"],
        mixed_precision = cfg["training"]["mixed_precision"],
        log_interval    = wandb_cfg.get("log_interval", 50),
        wandb_logger    = logger,
    )

    best = trainer.fit()
    print(f"\n[train] Done. Best val_auroc_macro = {best.get('val_auroc_macro', 0):.4f}")
    if logger.run_url:
        print(f"[train] WandB run: {logger.run_url}")


if __name__ == "__main__":
    main()
