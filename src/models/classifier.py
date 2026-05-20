"""
Unified model factory for DenseNet-121 and ViT-Base/16.

Both models are loaded from timm with ImageNet pre-trained weights and their
final classification head replaced to output `num_classes` logits.
"""

import timm
import torch
import torch.nn as nn


class CXRClassifier(nn.Module):
    """Thin wrapper around a timm backbone for multi-label CXR classification.

    Args:
        backbone_name: timm model name, e.g. 'densenet121' or 'vit_base_patch16_224'.
        num_classes:   Number of output logits (14 for NIH-14).
        pretrained:    Load ImageNet weights from timm.
        dropout:       Dropout probability inserted before the head (0 = disabled).
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # remove the default head
        )
        feature_dim = self.backbone.num_features

        head_layers: list[nn.Module] = []
        if dropout > 0.0:
            head_layers.append(nn.Dropout(p=dropout))
        head_layers.append(nn.Linear(feature_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def is_vit(self) -> bool:
        return "vit" in self.backbone_name.lower()


def build_model(
    name: str,
    num_classes: int = 14,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> CXRClassifier:
    """Construct and return a CXRClassifier.

    Args:
        name:        'densenet121' or 'vit_base_patch16_224'.
        num_classes: Output logit count.
        pretrained:  ImageNet init.
        dropout:     Head dropout probability.
    """
    supported = {"densenet121", "vit_base_patch16_224"}
    if name not in supported:
        raise ValueError(f"Model '{name}' not supported. Choose from: {supported}")
    return CXRClassifier(name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
