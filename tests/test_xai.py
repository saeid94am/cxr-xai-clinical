"""
Tests for src/xai and src/evaluation — XAI methods and quantitative metrics.

All tests run on CPU with untrained (pretrained=False) models and synthetic
random image tensors. We verify output shapes, value ranges, and basic
behavioural contracts — not numerical accuracy, which requires real data.
"""

import numpy as np
import pytest
import torch

from src.models import build_model
from src.xai import (
    compute_cam_batch,
    compute_integrated_gradients,
    compute_attention_rollout,
)
from src.evaluation import (
    compute_deletion_insertion,
    compute_spearman_stability,
    compute_sanity_check,
    compute_road,
)
from src.evaluation.pointing_game import pointing_game_hit


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def densenet():
    m = build_model("densenet121", num_classes=14, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def vit():
    m = build_model("vit_base_patch16_224", num_classes=14, pretrained=False)
    m.eval()
    return m


@pytest.fixture()
def batch2():
    """2-image batch of random normalised tensors."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


@pytest.fixture()
def class_indices():
    return [0, 1]


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

def test_gradcam_plus_plus_output_shape(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    assert heatmaps.shape == (2, 224, 224)


def test_gradcam_plus_plus_range(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0 + 1e-6


def test_gradcam_plus_plus_not_all_zeros(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    assert heatmaps.sum() > 0


# ── HiResCAM ─────────────────────────────────────────────────────────────────

def test_hirescam_output_shape(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("hirescam", densenet, batch2, class_indices)
    assert heatmaps.shape == (2, 224, 224)


def test_hirescam_range(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("hirescam", densenet, batch2, class_indices)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0 + 1e-6


def test_cam_dispatch_unknown_raises(densenet, batch2, class_indices):
    with pytest.raises(ValueError, match="Unknown CAM method"):
        compute_cam_batch("nonexistent", densenet, batch2, class_indices)


# ── Integrated Gradients ──────────────────────────────────────────────────────

def test_ig_output_shape_densenet(densenet, batch2, class_indices):
    heatmaps = compute_integrated_gradients(
        densenet, batch2, class_indices, device="cpu", n_steps=5
    )
    assert heatmaps.shape == (2, 224, 224)


def test_ig_output_shape_vit(vit, batch2, class_indices):
    heatmaps = compute_integrated_gradients(
        vit, batch2, class_indices, device="cpu", n_steps=5
    )
    assert heatmaps.shape == (2, 224, 224)


def test_ig_range(densenet, batch2, class_indices):
    heatmaps = compute_integrated_gradients(
        densenet, batch2, class_indices, device="cpu", n_steps=5
    )
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0 + 1e-6


def test_ig_gaussian_baseline(densenet, batch2, class_indices):
    heatmaps = compute_integrated_gradients(
        densenet, batch2, class_indices, device="cpu",
        n_steps=5, baseline_mode="gaussian",
    )
    assert heatmaps.shape == (2, 224, 224)


# ── Attention Rollout ─────────────────────────────────────────────────────────

def test_attention_rollout_output_shape(vit, batch2):
    heatmaps = compute_attention_rollout(vit, batch2, device="cpu")
    assert heatmaps.shape == (2, 224, 224)


def test_attention_rollout_range(vit, batch2):
    heatmaps = compute_attention_rollout(vit, batch2, device="cpu")
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0 + 1e-6


def test_attention_rollout_not_uniform(vit, batch2):
    """Rollout map should not be a flat constant — some spatial variation expected."""
    heatmaps = compute_attention_rollout(vit, batch2, device="cpu")
    for i in range(heatmaps.shape[0]):
        assert heatmaps[i].std() > 0, f"Attention rollout map {i} is spatially uniform"


# ── Pointing game ─────────────────────────────────────────────────────────────

def test_pointing_game_hit_inside():
    heatmap = np.zeros((224, 224))
    heatmap[50, 80] = 1.0        # peak inside bbox
    assert pointing_game_hit(heatmap, bbox=(70, 40, 30, 30)) is True


def test_pointing_game_miss_outside():
    heatmap = np.zeros((224, 224))
    heatmap[10, 10] = 1.0        # peak far outside bbox
    assert pointing_game_hit(heatmap, bbox=(70, 40, 30, 30)) is False


def test_pointing_game_tolerance():
    heatmap = np.zeros((224, 224))
    heatmap[39, 70] = 1.0        # 1 px above bbox top edge
    # Without tolerance: miss
    assert pointing_game_hit(heatmap, bbox=(70, 40, 30, 30), tolerance_px=0) is False
    # With tolerance=2: hit
    assert pointing_game_hit(heatmap, bbox=(70, 40, 30, 30), tolerance_px=2) is True


# ── Deletion / Insertion AUC ──────────────────────────────────────────────────

def test_deletion_insertion_shapes(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    del_auc, ins_auc = compute_deletion_insertion(
        densenet, batch2, heatmaps, class_indices, device="cpu", n_steps=3
    )
    assert del_auc.shape == (2,)
    assert ins_auc.shape == (2,)


def test_deletion_insertion_ranges(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    del_auc, ins_auc = compute_deletion_insertion(
        densenet, batch2, heatmaps, class_indices, device="cpu", n_steps=3
    )
    # AUC over sigmoid outputs ∈ [0,1] swept over [0,1] → AUC ∈ [0,1]
    assert np.all(del_auc >= 0) and np.all(del_auc <= 1)
    assert np.all(ins_auc >= 0) and np.all(ins_auc <= 1)


# ── Spearman stability ────────────────────────────────────────────────────────

def test_spearman_stability_shape(densenet, batch2, class_indices):
    def heatmap_fn(imgs):
        return compute_cam_batch("gradcam_plus_plus", densenet, imgs, class_indices)

    rho = compute_spearman_stability(heatmap_fn, batch2, noise_std=0.1, n_runs=2)
    assert rho.shape == (2,)


def test_spearman_stability_range(densenet, batch2, class_indices):
    def heatmap_fn(imgs):
        return compute_cam_batch("gradcam_plus_plus", densenet, imgs, class_indices)

    rho = compute_spearman_stability(heatmap_fn, batch2, noise_std=0.1, n_runs=2)
    assert np.all(rho >= -1.0) and np.all(rho <= 1.0)


# ── Sanity check ──────────────────────────────────────────────────────────────

def test_sanity_check_returns_expected_keys(densenet, batch2, class_indices):
    def factory(m):
        return lambda imgs: compute_cam_batch("gradcam_plus_plus", m, imgs, class_indices)

    result = compute_sanity_check(densenet, factory, batch2, class_indices)
    assert "rho_curve" in result
    assert "layer_names" in result
    assert "pass" in result
    assert "final_rho" in result


def test_sanity_check_rho_curve_starts_at_one(densenet, batch2, class_indices):
    def factory(m):
        return lambda imgs: compute_cam_batch("gradcam_plus_plus", m, imgs, class_indices)

    result = compute_sanity_check(densenet, factory, batch2, class_indices)
    assert result["rho_curve"][0] == pytest.approx(1.0)


# ── ROAD ─────────────────────────────────────────────────────────────────────

def test_road_output_shape(densenet, batch2, class_indices):
    heatmaps = compute_cam_batch("gradcam_plus_plus", densenet, batch2, class_indices)
    scores = compute_road(
        densenet, batch2, heatmaps, class_indices,
        device="cpu", percentages=[20, 50, 80],
    )
    assert scores.shape == (2,)
