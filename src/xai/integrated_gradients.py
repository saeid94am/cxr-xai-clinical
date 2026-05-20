"""
Integrated Gradients (IG) for ViT-Base/16 and DenseNet-121.

IG satisfies the completeness axiom: attribution values sum exactly to the
difference between the model output at the input and at the baseline.
This makes it theoretically grounded and model-agnostic — it can be applied
to both ViT and DenseNet, enabling a unified cross-architecture comparison.

Baseline: black image (zero tensor after ImageNet normalisation is the
zero-mean baseline; a Gaussian noise baseline is also supported).

Returns a (H, W) saliency map collapsed from the (3, H, W) attribution
by taking the absolute value and summing across the channel dimension,
then normalising to [0, 1].
"""

from __future__ import annotations

from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


def _make_baseline(
    images: torch.Tensor,
    mode: Literal["zero", "gaussian"],
    seed: int = 42,
) -> torch.Tensor:
    if mode == "gaussian":
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.randn_like(images, generator=rng) * 0.001
    return torch.zeros_like(images)


def _attributions_to_heatmap(attrs: torch.Tensor) -> np.ndarray:
    """Collapse (B, 3, H, W) attributions → (B, H, W) normalised heatmap."""
    # Sum absolute attributions across colour channels
    heatmap = attrs.abs().sum(dim=1)  # (B, H, W)
    heatmap = heatmap.cpu().float().numpy()

    # Normalise each image independently to [0, 1]
    b_min = heatmap.min(axis=(1, 2), keepdims=True)
    b_max = heatmap.max(axis=(1, 2), keepdims=True)
    heatmap = (heatmap - b_min) / (b_max - b_min + 1e-8)
    return heatmap


def compute_integrated_gradients(
    model: nn.Module,
    images: torch.Tensor,
    class_indices: List[int],
    device: str = "cpu",
    n_steps: int = 50,
    baseline_mode: Literal["zero", "gaussian"] = "zero",
) -> np.ndarray:
    """Compute Integrated Gradients heatmaps for a batch of images.

    Args:
        model:          CXRClassifier (ViT or DenseNet backbone).
        images:         Float tensor (B, 3, H, W), already normalised.
        class_indices:  Target class index per image (length B).
        device:         Device string ('cuda' or 'cpu').
        n_steps:        Number of IG interpolation steps (50 is standard).
        baseline_mode:  'zero' (black image) or 'gaussian' (near-zero noise).

    Returns:
        heatmaps: np.ndarray of shape (B, H, W) in [0, 1].
    """
    model.eval()
    images = images.to(device)
    baseline = _make_baseline(images, baseline_mode).to(device)

    ig = IntegratedGradients(model)
    # captum requires a scalar target per sample
    targets = torch.tensor(class_indices, dtype=torch.long, device=device)

    attrs = ig.attribute(
        inputs=images,
        baselines=baseline,
        target=targets,
        n_steps=n_steps,
        internal_batch_size=images.shape[0],
    )  # (B, 3, H, W)

    return _attributions_to_heatmap(attrs)
