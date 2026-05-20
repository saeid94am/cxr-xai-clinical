"""
Deletion AUC and Insertion AUC for XAI faithfulness evaluation.

Deletion AUC: progressively remove the most salient pixels (replace with
  blurred baseline) and measure how classification score drops.
  Lower AUC = more faithful (score collapses faster when important pixels removed).

Insertion AUC: progressively reveal the most salient pixels from a fully
  blurred baseline and measure how the score rises.
  Higher AUC = more faithful (score rises faster as important pixels added).

Both metrics sweep k = [0%, 10%, 20%, ..., 100%] of pixels (n_steps=10 by
default; configurable). The AUC is the area under the score-vs-k curve,
computed with the trapezoidal rule.

Implementation is custom (not pytorch-grad-cam's built-in) so we have full
control over the baseline and can run on 4 GB VRAM with small batch sizes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_blur_baseline(images: torch.Tensor, kernel_size: int = 51) -> torch.Tensor:
    """Return a heavily blurred version of images as the insertion baseline."""
    # Pad to avoid border artefacts, then apply average pool as cheap blur
    pad = kernel_size // 2
    blurred = F.avg_pool2d(
        F.pad(images, [pad] * 4, mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )
    return blurred


def _build_mask(heatmap: np.ndarray, fraction: float, top: bool) -> np.ndarray:
    """Return a boolean (H, W) mask of the top/bottom `fraction` of pixels.

    Args:
        heatmap:  (H, W) saliency map.
        fraction: Fraction in [0, 1] of pixels to select.
        top:      True = highest-saliency pixels; False = lowest.

    Returns:
        Boolean mask, True where pixels are selected.
    """
    n_pixels = heatmap.size
    k = max(1, int(fraction * n_pixels))
    flat = heatmap.ravel()
    if top:
        indices = np.argpartition(flat, -k)[-k:]
    else:
        indices = np.argpartition(flat, k)[:k]
    mask = np.zeros(n_pixels, dtype=bool)
    mask[indices] = True
    return mask.reshape(heatmap.shape)


def _score_at_fraction(
    model: nn.Module,
    images: torch.Tensor,  # (B, 3, H, W)
    baseline: torch.Tensor,  # (B, 3, H, W)
    heatmaps: np.ndarray,  # (B, H, W)
    class_indices: List[int],
    fraction: float,
    mode: str,  # 'deletion' | 'insertion'
    device: str,
) -> np.ndarray:
    """Return per-image sigmoid scores after masking fraction of pixels."""
    B, C, H, W = images.shape
    masked = images.clone()

    for i in range(B):
        mask = _build_mask(heatmaps[i], fraction, top=True)  # (H, W)
        mask_t = torch.from_numpy(mask).to(device)  # (H, W)

        if mode == "deletion":
            # Remove salient pixels → replace with baseline
            masked[i, :, mask_t] = baseline[i, :, mask_t]
        else:
            # Insertion: start from baseline, reveal salient pixels
            revealed = baseline[i].clone()
            revealed[:, mask_t] = images[i, :, mask_t]
            masked[i] = revealed

    with torch.no_grad():
        logits = model(masked.to(device))  # (B, num_classes)

    probs = torch.sigmoid(logits).cpu().numpy()  # (B, num_classes)
    scores = np.array([probs[i, c] for i, c in enumerate(class_indices)])
    return scores  # (B,)


def compute_deletion_insertion(
    model: nn.Module,
    images: torch.Tensor,
    heatmaps: np.ndarray,
    class_indices: List[int],
    device: str = "cpu",
    n_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Deletion AUC and Insertion AUC for a batch.

    Args:
        model:         CXRClassifier in eval mode.
        images:        Float tensor (B, 3, H, W), already normalised.
        heatmaps:      np.ndarray (B, H, W) saliency maps in [0, 1].
        class_indices: Target class index per image (length B).
        device:        Device string.
        n_steps:       Number of sweep steps (default 10 → 0%,10%,...,100%).

    Returns:
        deletion_auc:  np.ndarray (B,) — lower is more faithful.
        insertion_auc: np.ndarray (B,) — higher is more faithful.
    """
    model.eval()
    images = images.to(device)
    baseline = _gaussian_blur_baseline(images)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)

    del_scores = np.zeros((len(fractions), len(class_indices)))
    ins_scores = np.zeros((len(fractions), len(class_indices)))

    for step, frac in enumerate(fractions):
        del_scores[step] = _score_at_fraction(
            model, images, baseline, heatmaps, class_indices, frac, "deletion", device
        )
        ins_scores[step] = _score_at_fraction(
            model, images, baseline, heatmaps, class_indices, frac, "insertion", device
        )

    # AUC via trapezoidal rule over the fraction axis
    deletion_auc = np.trapz(del_scores, fractions, axis=0)  # (B,)
    insertion_auc = np.trapz(ins_scores, fractions, axis=0)  # (B,)

    return deletion_auc, insertion_auc
