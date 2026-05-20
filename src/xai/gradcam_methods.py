"""
Grad-CAM++ and HiResCAM for DenseNet-121 (CNN backbone).

Both methods use grad-cam (PyPI: grad-cam). The target layer is the last conv layer
of DenseNet-121: backbone.features.denseblock4 (last dense layer inside it).

Grad-CAM++: weights each pixel's gradient contribution per-class; sharper
             localization than vanilla Grad-CAM.
HiResCAM:   mathematically faithful — saliency values are guaranteed to
             reflect pixels the model provably used for its decision.
             Contrast with Grad-CAM++ is itself a publishable finding.

Both return a (H, W) numpy array in [0, 1] for a single image and class.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv layer of DenseNet-121 inside backbone.features."""
    # denseblock4 contains multiple _DenseLayer sub-modules; the last one's
    # conv2 is the final spatial feature map before global average pooling.
    denseblock4 = model.backbone.features.denseblock4
    last_denselayer = list(denseblock4.children())[-1]
    return last_denselayer.conv2


def compute_gradcam_plus_plus(
    model: nn.Module,
    images: torch.Tensor,
    class_indices: List[int],
    device: str = "cpu",
) -> np.ndarray:
    """Compute Grad-CAM++ heatmaps for a batch of images.

    Args:
        model:         CXRClassifier (DenseNet-121 backbone).
        images:        Float tensor (B, 3, H, W), already normalised.
        class_indices: List of length B — target class index per image.
        device:        Device string.

    Returns:
        heatmaps: np.ndarray of shape (B, H, W) in [0, 1].
    """
    model.eval()
    target_layer = _get_target_layer(model)
    targets = [ClassifierOutputTarget(c) for c in class_indices]

    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        heatmaps = cam(input_tensor=images.to(device), targets=targets)

    return heatmaps  # shape (B, H, W), float32 in [0,1]


def compute_hirescam(
    model: nn.Module,
    images: torch.Tensor,
    class_indices: List[int],
    device: str = "cpu",
) -> np.ndarray:
    """Compute HiResCAM heatmaps for a batch of images.

    Args:
        model:         CXRClassifier (DenseNet-121 backbone).
        images:        Float tensor (B, 3, H, W), already normalised.
        class_indices: List of length B — target class index per image.
        device:        Device string.

    Returns:
        heatmaps: np.ndarray of shape (B, H, W) in [0, 1].
    """
    model.eval()
    target_layer = _get_target_layer(model)
    targets = [ClassifierOutputTarget(c) for c in class_indices]

    with HiResCAM(model=model, target_layers=[target_layer]) as cam:
        heatmaps = cam(input_tensor=images.to(device), targets=targets)

    return heatmaps  # shape (B, H, W), float32 in [0,1]


def compute_cam_batch(
    method: str,
    model: nn.Module,
    images: torch.Tensor,
    class_indices: List[int],
    device: str = "cpu",
) -> np.ndarray:
    """Dispatch to Grad-CAM++ or HiResCAM by name.

    Args:
        method: 'gradcam_plus_plus' or 'hirescam'.
        model:  CXRClassifier with DenseNet-121 backbone.
        images: Float tensor (B, 3, H, W).
        class_indices: Target class per image.
        device: Device string.

    Returns:
        heatmaps: np.ndarray (B, H, W) in [0, 1].
    """
    if method == "gradcam_plus_plus":
        return compute_gradcam_plus_plus(model, images, class_indices, device)
    if method == "hirescam":
        return compute_hirescam(model, images, class_indices, device)
    raise ValueError(f"Unknown CAM method: '{method}'. Choose gradcam_plus_plus or hirescam.")
