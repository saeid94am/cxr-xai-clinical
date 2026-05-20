"""
Loss functions for multi-label CXR classification.

BCEWithLogitsLoss with per-class positive weights is the correct choice for
NIH-14: labels are binary, classes are heavily imbalanced (e.g., Hernia <0.2%),
and pos_weight shifts the decision boundary without resampling.
"""

import torch
import torch.nn as nn


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Return BCEWithLogitsLoss, optionally with per-class positive weights.

    Args:
        pos_weight: 1-D tensor of length num_classes. Each value is
                    neg_count / pos_count for that class, computed from the
                    training set via NIH14Dataset.compute_pos_weights().
                    Pass None to use unweighted loss (e.g. for CheXpert eval).
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
