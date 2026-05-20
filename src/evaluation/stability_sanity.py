"""
Spearman rank-correlation stability and cascading randomization sanity check.

────────────────────────────────────────────────────────────────────────────
Spearman ρ stability (Montavon et al. / Alvarez-Melis & Jaakkola 2018)
────────────────────────────────────────────────────────────────────────────
Measures how consistent a saliency method is under small input perturbations.
Protocol:
  1. Generate the original heatmap for an image.
  2. Add Gaussian noise (std=0.1 of input std) to the image n_runs times.
  3. Generate a heatmap for each noisy version.
  4. Compute Spearman rank correlation between the original and each noisy
     heatmap (both flattened to 1-D pixel vectors).
  5. Return the mean ρ across runs and images. Higher = more stable.

────────────────────────────────────────────────────────────────────────────
Cascading randomization sanity check (Adebayo et al. 2018)
────────────────────────────────────────────────────────────────────────────
Verifies that a saliency method is actually sensitive to model weights.
Protocol:
  1. Start with the trained model. Generate a heatmap → baseline.
  2. Randomly reinitialize the topmost layer's weights.
  3. Generate a new heatmap. Compute Spearman ρ with the baseline.
  4. Cascade: reinitialize the next layer down, repeat until all layers
     are randomized.
  5. A method PASSES if ρ decreases monotonically as more layers are
     randomized. A method that shows no change is not explaining the model.

Returns a pass/fail verdict and the ρ curve for plotting.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# ── Spearman stability ────────────────────────────────────────────────────────


def compute_spearman_stability(
    heatmap_fn: Callable[[torch.Tensor], np.ndarray],
    images: torch.Tensor,
    noise_std: float = 0.1,
    n_runs: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Compute mean Spearman ρ stability for a batch of images.

    Args:
        heatmap_fn: Callable that takes (B, 3, H, W) tensor and returns
                    (B, H, W) numpy heatmap. Wraps any XAI method.
        images:     Float tensor (B, 3, H, W), already normalised.
        noise_std:  Std of Gaussian noise added to each image.
        n_runs:     Number of noisy repetitions per image.
        seed:       Base random seed (incremented per run for reproducibility).

    Returns:
        rho: np.ndarray (B,) — mean Spearman ρ across n_runs, per image.
             Values in [-1, 1]; higher = more stable.
    """
    original_heatmaps = heatmap_fn(images)  # (B, H, W)
    B = images.shape[0]
    rho_runs = np.zeros((n_runs, B))

    for run in range(n_runs):
        rng = torch.Generator()
        rng.manual_seed(seed + run)
        noise = torch.randn_like(images, generator=rng) * noise_std
        noisy_images = (images + noise).clamp(-3.0, 3.0)  # stay in normalised range

        noisy_heatmaps = heatmap_fn(noisy_images)  # (B, H, W)

        for i in range(B):
            orig_flat = original_heatmaps[i].ravel()
            noisy_flat = noisy_heatmaps[i].ravel()
            rho, _ = spearmanr(orig_flat, noisy_flat)
            rho_runs[run, i] = rho if np.isfinite(rho) else 0.0

    return rho_runs.mean(axis=0)  # (B,)


# ── Cascading randomization sanity check ─────────────────────────────────────


def _get_cascade_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return named layers in reverse order (output→input) for cascading reset.

    Skips BatchNorm and bias-only layers — resetting those alone has no
    meaningful effect on the saliency map (Adebayo et al. 2018).
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append((name, module))
    return list(reversed(layers))  # outer (head) first


def _reset_layer_weights(module: nn.Module, seed: int) -> None:
    """Reinitialize a layer's weights with Kaiming uniform (matches timm init)."""
    torch.manual_seed(seed)
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, a=0)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def compute_sanity_check(
    model: nn.Module,
    heatmap_fn_factory: Callable[[nn.Module], Callable[[torch.Tensor], np.ndarray]],
    images: torch.Tensor,
    class_indices: List[int],
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the cascading randomization sanity check on a batch of images.

    Args:
        model:             Trained CXRClassifier (will be deep-copied; original
                           weights are NOT modified).
        heatmap_fn_factory: Callable that takes a model and returns a heatmap_fn.
                            This allows swapping the model while keeping XAI config.
        images:            Float tensor (B, 3, H, W).
        class_indices:     Target class per image (length B).
        seed:              Base seed for weight randomization.

    Returns:
        Dict with keys:
          'rho_curve'   — list of mean Spearman ρ values, one per cascade step
                          (step 0 = original model; step N = fully randomized)
          'layer_names' — list of layer names reset at each step
          'pass'        — bool: True if ρ decreases monotonically (method passes)
          'final_rho'   — ρ after all layers randomized (should be near 0 for pass)
    """
    rand_model = copy.deepcopy(model)
    rand_model.eval()

    # Baseline heatmaps from the trained model
    baseline_fn = heatmap_fn_factory(rand_model)
    baseline_heatmaps = baseline_fn(images)  # (B, H, W)

    cascade_layers = _get_cascade_layers(rand_model)
    rho_curve: List[float] = []
    layer_names: List[str] = []

    # Step 0: correlation with itself = 1.0 (sanity baseline)
    rho_curve.append(1.0)
    layer_names.append("(original)")

    for step, (name, module) in enumerate(cascade_layers):
        _reset_layer_weights(module, seed=seed + step)

        current_fn = heatmap_fn_factory(rand_model)
        current_heatmaps = current_fn(images)  # (B, H, W)

        rhos = []
        for i in range(images.shape[0]):
            rho, _ = spearmanr(baseline_heatmaps[i].ravel(), current_heatmaps[i].ravel())
            rhos.append(rho if np.isfinite(rho) else 0.0)

        mean_rho = float(np.mean(rhos))
        rho_curve.append(mean_rho)
        layer_names.append(name)

    # Pass criterion: final ρ is substantially below the initial ρ
    final_rho = rho_curve[-1]
    # Monotonically decreasing: each step's ρ ≤ previous (allow small noise ≤ 0.02)
    diffs = np.diff(rho_curve)
    is_monotone = bool(np.all(diffs <= 0.02))
    passes = is_monotone and (final_rho < 0.5)

    return {
        "rho_curve": rho_curve,
        "layer_names": layer_names,
        "pass": passes,
        "final_rho": final_rho,
    }
