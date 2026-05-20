"""
Generate and save XAI heatmaps for a trained model on the NIH-14 test set.

Usage:
    python scripts/generate_xai.py \\
        --config configs/train.yaml \\
        --xai-config configs/xai.yaml \\
        --checkpoint results/checkpoints/best_densenet121.pt \\
        --model densenet121 \\
        --split test          # test | val
        --max-images 500      # limit for quick runs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import NIH14_CLASSES, NIH14Dataset, val_transforms
from src.models import build_model
from src.training import load_checkpoint
from src.utils import set_seed
from src.xai import (
    compute_attention_rollout,
    compute_cam_batch,
    compute_integrated_gradients,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate XAI heatmaps")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--xai-config", default="configs/xai.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_<model>.pt")
    p.add_argument("--model", required=True, choices=["densenet121", "vit_base_patch16_224"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap number of images (useful for quick checks)",
    )
    return p.parse_args()


def _xai_methods_for(model_name: str, xai_cfg: dict) -> list[str]:
    return xai_cfg["methods"].get(model_name, [])


def _compute_heatmap(
    method: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    class_indices: list[int],
    xai_cfg: dict,
    device: str,
) -> np.ndarray:
    if method in ("gradcam_plus_plus", "hirescam"):
        return compute_cam_batch(method, model, images, class_indices, device)
    if method == "integrated_gradients":
        ig_cfg = xai_cfg.get("integrated_gradients", {})
        return compute_integrated_gradients(
            model,
            images,
            class_indices,
            device,
            n_steps=ig_cfg.get("n_steps", 50),
            baseline_mode=ig_cfg.get("baseline", "zero"),
        )
    if method == "attention_rollout":
        return compute_attention_rollout(model, images, device)
    raise ValueError(f"Unknown XAI method: {method}")


def _save_heatmap(heatmap: np.ndarray, out_path: Path) -> None:
    """Save a (H, W) float32 [0,1] array as a grayscale PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    img.save(out_path)


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.xai_config) as f:
        xai_cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = args.model
    figures_dir = Path(cfg["figures_dir"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], pretrained=False).to(
        device
    )
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    split_file = (
        cfg["data"]["nih14_test_list"]
        if args.split == "test"
        else cfg["data"]["nih14_train_val_list"]
    )
    dataset = NIH14Dataset(
        image_dir=cfg["data"]["nih14_images"],
        labels_csv=cfg["data"]["nih14_labels"],
        split_file=split_file,
        transform=val_transforms(cfg["input"]["val_size"]),
    )

    methods = _xai_methods_for(model_name, xai_cfg)
    if not methods:
        print(f"[generate_xai] No XAI methods configured for {model_name}. Check xai.yaml.")
        return

    print(
        f"[generate_xai] model={model_name}  split={args.split}  "
        f"methods={methods}  n={len(dataset)}"
    )

    # ── Generate per-image, per-class ─────────────────────────────────────────
    max_images = args.max_images or len(dataset)

    for img_idx in tqdm(range(min(max_images, len(dataset))), desc="images"):
        image, label, img_path = dataset[img_idx]
        image_batch = image.unsqueeze(0)  # (1, 3, H, W)

        # Generate one heatmap per positive class (or all classes if no positives)
        pos_classes = label.nonzero(as_tuple=True)[0].tolist()
        target_classes = pos_classes if pos_classes else list(range(len(NIH14_CLASSES)))

        stem = Path(img_path).stem

        for cls_idx in target_classes:
            cls_name = NIH14_CLASSES[cls_idx]
            for method in methods:
                heatmap = _compute_heatmap(method, model, image_batch, [cls_idx], xai_cfg, device)[
                    0
                ]  # (H, W)

                out_path = figures_dir / model_name / method / cls_name / f"{stem}.png"
                if xai_cfg["output"].get("save_heatmaps", True):
                    _save_heatmap(heatmap, out_path)

    print(f"[generate_xai] Done. Heatmaps saved to {figures_dir}/")


if __name__ == "__main__":
    main()
