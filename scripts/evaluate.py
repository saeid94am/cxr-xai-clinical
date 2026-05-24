"""
Run the full quantitative XAI evaluation pipeline on a trained model.

Computes all six metrics for all XAI methods configured for the given model,
then writes results/metrics/xai_comparison_table.csv.

Usage:
    python scripts/evaluate.py \\
        --config configs/train.yaml \\
        --xai-config configs/xai.yaml \\
        --checkpoint results/checkpoints/best_densenet121.pt \\
        --model densenet121 \\
        --max-images 200
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import NIH14_CLASSES, NIH14Dataset, val_transforms
from src.evaluation import (
    build_summary_row,
    compute_deletion_insertion,
    compute_pointing_game,
    compute_road,
    compute_sanity_check,
    compute_spearman_stability,
    save_summary_table,
)
from src.models import build_model
from src.training import load_checkpoint
from src.utils import set_seed
from src.xai import (
    compute_attention_rollout,
    compute_cam_batch,
    compute_integrated_gradients,
)

METHOD_DISPLAY = {
    "gradcam_plus_plus": "Grad-CAM++",
    "hirescam": "HiResCAM",
    "integrated_gradients": "Integrated Gradients",
    "attention_rollout": "Attention Rollout",
}

BACKBONE_DISPLAY = {
    "densenet121": "DenseNet-121",
    "vit_base_patch16_224": "ViT-Base/16",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantitative XAI evaluation")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--xai-config", default="configs/xai.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=["densenet121", "vit_base_patch16_224"])
    p.add_argument(
        "--max-images", type=int, default=200, help="Images to evaluate (default 200 for speed)"
    )
    return p.parse_args()


def _make_heatmap_fn(
    method: str,
    model: torch.nn.Module,
    class_indices: list[int],
    xai_cfg: dict,
    device: str,
) -> Callable[[torch.Tensor], np.ndarray]:
    """Return a heatmap_fn(images) → (B,H,W) ndarray closure."""

    def fn(images: torch.Tensor) -> np.ndarray:
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
        raise ValueError(f"Unknown method: {method}")

    return fn


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.xai_config) as f:
        xai_cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = args.model
    methods: list[str] = xai_cfg["methods"].get(model_name, [])

    if not methods:
        print(f"No XAI methods configured for {model_name}. Check xai.yaml.")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], pretrained=False).to(
        device
    )
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Dataset (test split) ──────────────────────────────────────────────────
    dataset = NIH14Dataset(
        image_dir=cfg["data"]["nih14_images"],
        labels_csv=cfg["data"]["nih14_labels"],
        split_file=cfg["data"]["nih14_test_list"],
        transform=val_transforms(cfg["input"]["val_size"]),
    )
    n = min(args.max_images, len(dataset))
    print(f"[evaluate] model={model_name}  methods={methods}  n_images={n}")

    # Collect images and labels as a single batch for metric computation
    images_list, labels_list, fnames = [], [], []
    for i in range(n):
        img, lbl, path = dataset[i]
        images_list.append(img)
        labels_list.append(lbl)
        fnames.append(Path(path).name)

    images = torch.stack(images_list)  # (N, 3, H, W)

    # Use the first positive class per image as target (fallback: class 0)
    class_indices = [
        int(lbl.nonzero(as_tuple=True)[0][0]) if lbl.sum() > 0 else 0 for lbl in labels_list
    ]

    summary_rows = []

    for method in methods:
        print(f"\n[evaluate] ── {METHOD_DISPLAY[method]} ──")
        heatmap_fn = _make_heatmap_fn(method, model, class_indices, xai_cfg, device)

        # ── Generate heatmaps ─────────────────────────────────────────────────
        print("  Generating heatmaps...")
        heatmaps = heatmap_fn(images)  # (N, H, W)

        # ── Pointing game ─────────────────────────────────────────────────────
        print("  Pointing game...")
        heatmap_dict = {fnames[i]: {NIH14_CLASSES[class_indices[i]]: heatmaps[i]} for i in range(n)}
        pg_df = compute_pointing_game(
            heatmap_dict,
            cfg["data"]["nih14_bbox"],
            img_size=cfg["input"]["val_size"],
            tolerance_px=xai_cfg["pointing_game"].get("tolerance_px", 0),
        )
        overall_row = pg_df[pg_df["label"] == "OVERALL"]
        pg_acc = float(overall_row["accuracy"].values[0]) if not overall_row.empty else float("nan")
        print(f"    Pointing game accuracy: {pg_acc:.4f}")

        # ── Deletion / Insertion AUC ──────────────────────────────────────────
        print("  Deletion / Insertion AUC...")
        di_cfg = xai_cfg.get("deletion_insertion", {})
        del_auc, ins_auc = compute_deletion_insertion(
            model,
            images,
            heatmaps,
            class_indices,
            device,
            n_steps=di_cfg.get("n_steps", 10),
        )
        print(f"    Deletion AUC:  {del_auc.mean():.4f}  Insertion AUC: {ins_auc.mean():.4f}")

        # ── Spearman stability ────────────────────────────────────────────────
        print("  Spearman stability...")
        stab_cfg = xai_cfg.get("stability", {})
        rho = compute_spearman_stability(
            heatmap_fn,
            images,
            noise_std=stab_cfg.get("noise_std", 0.1),
            n_runs=stab_cfg.get("n_runs", 3),
        )
        print(f"    Spearman ρ: {rho.mean():.4f}")

        # ── ROAD ──────────────────────────────────────────────────────────────
        print("  ROAD faithfulness...")
        road_pcts = xai_cfg.get("road", {}).get("percentages", None)
        road_scores = compute_road(
            model,
            images,
            heatmaps,
            class_indices,
            device,
            percentages=road_pcts,
        )
        print(f"    ROAD: {road_scores.mean():.4f}")

        # ── Sanity check ──────────────────────────────────────────────────────
        print("  Sanity check (cascading randomization)...")
        # Use a small subset (first 8 images) to keep runtime reasonable
        subset_imgs = images[:8]
        subset_classes = class_indices[:8]

        def factory(m):
            return _make_heatmap_fn(method, m, subset_classes, xai_cfg, device)

        sanity = compute_sanity_check(model, factory, subset_imgs, subset_classes)
        verdict = "Pass" if sanity["pass"] else "Fail"
        print(f"    Sanity check: {verdict}  (final ρ={sanity['final_rho']:.4f})")

        summary_rows.append(
            build_summary_row(
                method=METHOD_DISPLAY[method],
                backbone=BACKBONE_DISPLAY[model_name],
                pointing_game=pg_acc,
                deletion_auc=float(del_auc.mean()),
                insertion_auc=float(ins_auc.mean()),
                spearman_rho=float(rho.mean()),
                road=float(road_scores.mean()),
                sanity_pass=sanity["pass"],
            )
        )

    # ── Save summary table ────────────────────────────────────────────────────
    out_path = Path(cfg["metrics_dir"]) / "xai_comparison_table.csv"
    df = save_summary_table(summary_rows, str(out_path))
    print("\n[evaluate] Done.\n")
    print(df.to_string(index=False))

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb

            run = wandb.init(
                project=wandb_cfg.get("project", "cxr-xai-clinical"),
                entity=wandb_cfg.get("entity"),
                name=f"xai_eval_{model_name}",
                job_type="xai_evaluation",
                config={"model": model_name, "n_images": n, "methods": methods},
                dir="/tmp",
            )
            wandb.log({"xai_comparison": wandb.Table(dataframe=df)})
            for _, row in df.iterrows():
                method_key = (
                    row["XAI Method"].lower().replace(" ", "_").replace("/", "_").replace("+", "p")
                )
                row_metrics = {
                    f"xai/{method_key}/pointing_game": row["Pointing Game ↑"],
                    f"xai/{method_key}/deletion_auc": row["Deletion AUC ↓"],
                    f"xai/{method_key}/insertion_auc": row["Insertion AUC ↑"],
                    f"xai/{method_key}/spearman_rho": row["Spearman ρ ↑"],
                    f"xai/{method_key}/road": row["ROAD ↑"],
                    f"xai/{method_key}/sanity_pass": 1.0 if row["Sanity Check"] == "Pass" else 0.0,
                }
                wandb.log(row_metrics)
            artifact = wandb.Artifact(f"xai_results_{model_name}", type="evaluation")
            artifact.add_file(str(out_path))
            run.log_artifact(artifact)
            run.finish()
            print("[evaluate] Results logged to WandB.")
        except Exception as e:
            print(f"[evaluate] WandB logging failed (results saved locally): {e}")


if __name__ == "__main__":
    main()
