"""
Layer-wise learning rate decay for DenseNet-121 and ViT-Base/16.

Outer (later) layers train at the base lr; deeper layers are scaled down
by `decay_factor` per group, so the backbone bottom trains slowest.
This is the standard LLRD approach used in CheXNet fine-tuning literature.
"""

from typing import List, Dict, Any

import torch.nn as nn

from .classifier import CXRClassifier


def get_layerwise_param_groups(
    model: CXRClassifier,
    base_lr: float,
    decay_factor: float = 0.9,
    weight_decay: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Return AdamW param groups with per-layer learning rates.

    Groups (outer → inner, lr high → low):
      DenseNet-121: head → denseblock4 → transition3 → denseblock3 → ... → features.conv0
      ViT-Base/16:  head → blocks[11] → blocks[10] → ... → blocks[0] → patch_embed

    Args:
        model:        Trained CXRClassifier instance.
        base_lr:      Learning rate for the outermost group (head).
        decay_factor: Multiplicative decay per group moving inward.
        weight_decay: Weight decay applied to all groups.

    Returns:
        List of dicts suitable for passing directly to torch.optim.AdamW.
    """
    if model.is_vit():
        return _vit_param_groups(model, base_lr, decay_factor, weight_decay)
    return _densenet_param_groups(model, base_lr, decay_factor, weight_decay)


def _densenet_param_groups(
    model: CXRClassifier,
    base_lr: float,
    decay: float,
    wd: float,
) -> List[Dict[str, Any]]:
    # Layer groups ordered outer → inner
    groups_names = [
        ["head"],
        ["backbone.features.denseblock4", "backbone.features.norm5"],
        ["backbone.features.transition3"],
        ["backbone.features.denseblock3"],
        ["backbone.features.transition2"],
        ["backbone.features.denseblock2"],
        ["backbone.features.transition1"],
        ["backbone.features.denseblock1"],
        ["backbone.features.conv0", "backbone.features.norm0"],
    ]
    return _build_groups(model, groups_names, base_lr, decay, wd)


def _vit_param_groups(
    model: CXRClassifier,
    base_lr: float,
    decay: float,
    wd: float,
) -> List[Dict[str, Any]]:
    # ViT has 12 transformer blocks (0–11); group them outer→inner
    groups_names = [["head"]]
    for block_idx in range(11, -1, -1):
        groups_names.append([f"backbone.blocks.{block_idx}"])
    groups_names.append(["backbone.patch_embed", "backbone.cls_token", "backbone.pos_embed"])
    return _build_groups(model, groups_names, base_lr, decay, wd)


def _build_groups(
    model: CXRClassifier,
    groups_names: List[List[str]],
    base_lr: float,
    decay: float,
    wd: float,
) -> List[Dict[str, Any]]:
    assigned: set = set()
    param_groups: List[Dict[str, Any]] = []

    for g_idx, prefixes in enumerate(groups_names):
        lr = base_lr * (decay ** g_idx)
        params = []
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in prefixes) and name not in assigned:
                params.append(param)
                assigned.add(name)
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": wd})

    # Catch any remaining parameters (e.g., batch norm biases) at the lowest lr
    remaining = [
        p for n, p in model.named_parameters() if n not in assigned
    ]
    if remaining:
        lr = base_lr * (decay ** len(groups_names))
        param_groups.append({"params": remaining, "lr": lr, "weight_decay": wd})

    return param_groups
