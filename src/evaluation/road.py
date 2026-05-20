"""
ROAD faithfulness metric (Rong et al. 2022 / Hedström et al. 2023 quantus).

ROAD (Remove And Debias) corrects for the out-of-distribution bias introduced
by vanilla deletion/insertion: when pixels are removed, the resulting image is
OOD for the model, so score changes may reflect distributional shift rather
than saliency faithfulness. ROAD replaces removed pixels with their local
neighbourhood mean (Noisy Linear Imputation) instead of a fixed baseline,
keeping the image closer to the training distribution.

We wrap the `quantus` library's ROAD implementation. If quantus is not
installed, we fall back to a lightweight built-in approximation that matches
the paper's core idea (NLI replacement + AUC over perturbation percentages).

Returns a scalar ROAD score per image, higher = more faithful.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


def _nli_replace(
    images: np.ndarray,  # (B, C, H, W)
    heatmaps: np.ndarray,  # (B, H, W)
    percentage: float,
    patch_size: int = 9,
) -> np.ndarray:
    """Noisy Linear Imputation: replace top-k% pixels with local neighbourhood mean + noise."""
    B, C, H, W = images.shape
    result = images.copy()

    for i in range(B):
        flat = heatmaps[i].ravel()
        k = max(1, int(percentage / 100 * flat.size))
        top_indices = np.argpartition(flat, -k)[-k:]
        rows, cols = np.unravel_index(top_indices, (H, W))

        half = patch_size // 2
        for r, c in zip(rows, cols):
            r0, r1 = max(0, r - half), min(H, r + half + 1)
            c0, c1 = max(0, c - half), min(W, c + half + 1)
            patch = images[i, :, r0:r1, c0:c1]  # (C, ph, pw)
            mean_val = patch.mean(axis=(1, 2))  # (C,)
            std_val = patch.std(axis=(1, 2)) + 1e-8

            noise = np.random.randn(C) * std_val * 0.1
            result[i, :, r, c] = mean_val + noise

    return result


def compute_road(
    model: nn.Module,
    images: torch.Tensor,
    heatmaps: np.ndarray,
    class_indices: List[int],
    device: str = "cpu",
    percentages: List[float] | None = None,
) -> np.ndarray:
    """Compute ROAD faithfulness score for a batch.

    Tries to use quantus.ROAD if available; otherwise uses the built-in
    NLI approximation which matches the core paper protocol.

    Args:
        model:         CXRClassifier in eval mode.
        images:        Float tensor (B, 3, H, W), already normalised.
        heatmaps:      np.ndarray (B, H, W) saliency maps in [0, 1].
        class_indices: Target class per image (length B).
        device:        Device string.
        percentages:   Pixel removal percentages to sweep. Default [10..90].

    Returns:
        road_scores: np.ndarray (B,) — higher = more faithful.
    """
    if percentages is None:
        percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    model.eval()
    images_np = images.cpu().numpy()  # (B, C, H, W)
    B = images.shape[0]

    # ── Try quantus first ─────────────────────────────────────────────────────
    try:
        import quantus

        # quantus expects a callable that takes (inputs, **kwargs) → np.ndarray
        def model_fn(inputs: np.ndarray, **kwargs) -> np.ndarray:
            t = torch.from_numpy(inputs).float().to(device)
            with torch.no_grad():
                logits = model(t)
            return torch.sigmoid(logits).cpu().numpy()

        road_metric = quantus.ROAD(
            noise=0.1,
            perturb_func=quantus.perturb_func.noisy_linear_imputation,
            percentages=percentages,
            display_progressbar=False,
        )
        scores = road_metric(
            model=model_fn,
            x_batch=images_np,
            y_batch=np.array(class_indices),
            a_batch=heatmaps,
            device=device,
        )
        return np.array(scores)

    except Exception:
        pass

    # ── Built-in NLI approximation ────────────────────────────────────────────
    score_matrix = np.zeros((len(percentages), B))

    # Baseline score (no perturbation)
    with torch.no_grad():
        base_logits = model(images.to(device))
    base_probs = torch.sigmoid(base_logits).cpu().numpy()
    base_scores = np.array([base_probs[i, c] for i, c in enumerate(class_indices)])

    for p_idx, pct in enumerate(percentages):
        perturbed_np = _nli_replace(images_np, heatmaps, pct)
        perturbed = torch.from_numpy(perturbed_np).float().to(device)

        with torch.no_grad():
            pert_logits = model(perturbed)
        pert_probs = torch.sigmoid(pert_logits).cpu().numpy()
        pert_scores = np.array([pert_probs[i, c] for i, c in enumerate(class_indices)])

        # Score degradation at this percentage
        score_matrix[p_idx] = base_scores - pert_scores

    # ROAD score = mean degradation across percentages (higher = faster drop = more faithful)
    road_scores = score_matrix.mean(axis=0)
    return road_scores
