"""
Evaluate a trained model on the NIH-14 held-out test set.

Computes per-class AUROC and macro-AUROC, then writes
results/metrics/test_auroc_<model>.csv.

Usage:
    python scripts/test.py \
        --config configs/train_kaggle.yaml \
        --checkpoint results/checkpoints/best_densenet121.pt \
        --model densenet121
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import NIH14_CLASSES, NIH14Dataset, val_transforms
from src.models import build_model
from src.training import load_checkpoint
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-set classification evaluation")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_<model>.pt")
    p.add_argument("--model", required=True, choices=["densenet121", "vit_base_patch16_224"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = args.model

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], pretrained=False).to(
        device
    )
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Test dataset ──────────────────────────────────────────────────────────
    dataset = NIH14Dataset(
        image_dir=cfg["data"]["nih14_images"],
        labels_csv=cfg["data"]["nih14_labels"],
        split_file=cfg["data"]["nih14_test_list"],
        transform=val_transforms(cfg["input"]["val_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    print(f"[test] model={model_name}  test_images={len(dataset)}  device={device}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Inference", unit="batch"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs_np = np.concatenate(all_probs, axis=0)   # (N, 14)
    labels_np = np.concatenate(all_labels, axis=0)  # (N, 14)

    # ── Per-class AUROC ───────────────────────────────────────────────────────
    rows = []
    for i, cls_name in enumerate(NIH14_CLASSES):
        if labels_np[:, i].sum() == 0:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(labels_np[:, i], probs_np[:, i])
        n_pos = int(labels_np[:, i].sum())
        rows.append({"class": cls_name, "auroc": auroc, "n_positive": n_pos})

    valid_aurocs = [r["auroc"] for r in rows if not np.isnan(r["auroc"])]
    macro_auroc = float(np.mean(valid_aurocs))

    rows.append({"class": "MACRO_AVG", "auroc": macro_auroc, "n_positive": int(labels_np.sum())})

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_dir = Path(cfg["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"test_auroc_{model_name}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n[test] Results saved to {out_path}\n")
    print(df.to_string(index=False))
    print(f"\n[test] Macro-AUROC: {macro_auroc:.4f}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb

            run = wandb.init(
                project=wandb_cfg.get("project", "cxr-xai-clinical"),
                entity=wandb_cfg.get("entity"),
                name=f"test_{model_name}",
                job_type="test",
                config={"model": model_name, "n_test": len(dataset)},
                dir="/tmp",
            )
            log_dict = {}
            for r in rows:
                if r["class"] != "MACRO_AVG" and not np.isnan(r["auroc"]):
                    log_dict[f"test/{r['class']}"] = r["auroc"]
            log_dict["test/macro_auroc"] = macro_auroc
            wandb.log(log_dict)
            artifact = wandb.Artifact(f"test_results_{model_name}", type="evaluation")
            artifact.add_file(str(out_path))
            run.log_artifact(artifact)
            run.finish()
            print("[test] Results logged to WandB.")
        except Exception as e:
            print(f"[test] WandB logging failed (results saved locally): {e}")


if __name__ == "__main__":
    main()
