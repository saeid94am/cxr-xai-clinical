"""
Attention Rollout for ViT-Base/16 (Abnar & Zuidema, 2020).

Attention Rollout propagates attention weights across all transformer layers
by recursively multiplying attention matrices, accounting for residual
connections (identity added at each layer). This gives a patch-level
importance map that directly overlays on the CXR.

Key properties:
  - Computationally free: uses attention weights already computed in the
    forward pass — no extra gradient computation needed.
  - Produces patch-level (14×14 for ViT-Base/16 at 224px) heatmaps that
    are bilinearly upsampled to the full image resolution.
  - The [CLS] token row of the final rolled-out attention matrix is used
    as the saliency signal, since CLS attends to the patches that matter
    most for the classification decision.

Implementation registers forward hooks on all attention blocks to collect
the raw attention weight tensors, then removes the hooks after the forward
pass to leave the model clean.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionRollout:
    """Compute Attention Rollout heatmaps for a ViT model.

    Args:
        model:            CXRClassifier with ViT-Base/16 backbone.
        discard_ratio:    Fraction of lowest attention weights to zero out
                          before rollout (reduces noise). Default 0.9.
        head_fusion:      How to fuse multi-head attention: 'mean' | 'max' | 'min'.
    """

    def __init__(
        self,
        model: nn.Module,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
    ) -> None:
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self._attention_maps: List[torch.Tensor] = []
        self._hooks: list = []

    # ── Hook management ───────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Attach a forward hook to every attention block in the ViT."""
        self._attention_maps = []
        self._hooks = []

        for block in self.model.backbone.blocks:
            hook = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _attn_hook(self, module, input, output) -> None:
        # timm ViT attention modules expose .attn_weights after forward
        # when attn_drop is applied; we recompute from qkv directly.
        # timm stores the last attention map in module.attn_drop input if
        # we use the fused path; safest is to recompute via module.scale.
        # For timm>=0.9 the Attention module returns (x, attn_weights) when
        # return_attention=True is set — but we can't set that per-call here.
        # Instead, hook the softmax output via the module's forward internals.
        # We use a lightweight re-implementation of the attention computation.
        B, N, C = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        scale = module.scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # (B, heads, N, N)
        self._attention_maps.append(attn.detach().cpu())

    # ── Rollout ───────────────────────────────────────────────────────────────

    def _rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """Recursively fuse attention maps across layers.

        Args:
            attentions: List of (B, heads, N, N) tensors, one per layer.

        Returns:
            result: (B, N, N) rolled-out attention matrix.
        """
        B, _, N, _ = attentions[0].shape

        # Identity matrix for residual connection
        eye = torch.eye(N).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        result = eye.clone()
        for attn in attentions:
            # Fuse heads
            if self.head_fusion == "mean":
                fused = attn.mean(dim=1)  # (B, N, N)
            elif self.head_fusion == "max":
                fused = attn.max(dim=1).values
            else:
                fused = attn.min(dim=1).values

            # Discard lowest attention weights (noise reduction)
            flat = fused.view(B, -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
            threshold = threshold.unsqueeze(-1)  # (B, 1, 1)
            fused = torch.where(fused >= threshold, fused, torch.zeros_like(fused))

            # Add residual and normalise rows
            fused = fused + eye
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-8)

            result = torch.bmm(fused, result)

        return result  # (B, N, N)

    # ── Public API ────────────────────────────────────────────────────────────

    def __call__(
        self,
        images: torch.Tensor,
        device: str = "cpu",
    ) -> np.ndarray:
        """Compute Attention Rollout heatmaps for a batch.

        Args:
            images: Float tensor (B, 3, H, W), already normalised.
            device: Device string.

        Returns:
            heatmaps: np.ndarray of shape (B, H, W) in [0, 1].
        """
        self.model.eval()
        images = images.to(device)
        B, _, H, W = images.shape

        self._register_hooks()
        with torch.no_grad():
            _ = self.model(images)
        self._remove_hooks()

        rolled = self._rollout(self._attention_maps)  # (B, N+1, N+1) incl. CLS

        # CLS token is index 0; its attention to all patch tokens is row 0
        cls_attention = rolled[:, 0, 1:]  # (B, num_patches)

        # Reshape to spatial grid: ViT-Base/16 at 224px → 14×14 patches
        grid_size = int(cls_attention.shape[1] ** 0.5)
        patch_map = cls_attention.reshape(B, grid_size, grid_size)  # (B, 14, 14)

        # Upsample to full image resolution
        patch_map = patch_map.unsqueeze(1).float()  # (B, 1, 14, 14)
        heatmap = F.interpolate(patch_map, size=(H, W), mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze(1).numpy()  # (B, H, W)

        # Normalise each image to [0, 1]
        b_min = heatmap.min(axis=(1, 2), keepdims=True)
        b_max = heatmap.max(axis=(1, 2), keepdims=True)
        heatmap = (heatmap - b_min) / (b_max - b_min + 1e-8)

        return heatmap


def compute_attention_rollout(
    model: nn.Module,
    images: torch.Tensor,
    device: str = "cpu",
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Functional wrapper around AttentionRollout.

    Args:
        model:          CXRClassifier with ViT-Base/16 backbone.
        images:         Float tensor (B, 3, H, W).
        device:         Device string.
        discard_ratio:  Fraction of lowest-weight attention entries zeroed out.
        head_fusion:    'mean' | 'max' | 'min' across heads.

    Returns:
        heatmaps: np.ndarray (B, H, W) in [0, 1].
    """
    roller = AttentionRollout(model, discard_ratio=discard_ratio, head_fusion=head_fusion)
    return roller(images, device=device)
