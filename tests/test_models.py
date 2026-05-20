"""
Tests for src/models — CXRClassifier, build_model, and layer-wise LR decay.

Uses pretrained=False throughout so tests run without network access in CI.
A small 2-image batch of random tensors is sufficient to verify shapes and
parameter group structure.
"""

import pytest
import torch

from src.models import CXRClassifier, build_model, get_layerwise_param_groups


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def densenet():
    return build_model("densenet121", num_classes=14, pretrained=False)


@pytest.fixture(scope="module")
def vit():
    return build_model("vit_base_patch16_224", num_classes=14, pretrained=False)


@pytest.fixture()
def dummy_batch():
    """(2, 3, 224, 224) random float tensor — stands in for real CXR images."""
    return torch.randn(2, 3, 224, 224)


# ── build_model ───────────────────────────────────────────────────────────────

def test_build_model_densenet_returns_classifier():
    m = build_model("densenet121", pretrained=False)
    assert isinstance(m, CXRClassifier)


def test_build_model_vit_returns_classifier():
    m = build_model("vit_base_patch16_224", pretrained=False)
    assert isinstance(m, CXRClassifier)


def test_build_model_unsupported_raises():
    with pytest.raises(ValueError, match="not supported"):
        build_model("resnet50", pretrained=False)


# ── CXRClassifier — DenseNet-121 ──────────────────────────────────────────────

def test_densenet_output_shape(densenet, dummy_batch):
    densenet.eval()
    with torch.no_grad():
        out = densenet(dummy_batch)
    assert out.shape == (2, 14)


def test_densenet_output_is_logits(densenet, dummy_batch):
    """Output should be raw logits — values outside [0,1] are expected."""
    densenet.eval()
    with torch.no_grad():
        out = densenet(dummy_batch)
    # At least some logits should fall outside [0,1] for random weights
    assert not torch.all((out >= 0) & (out <= 1))


def test_densenet_is_vit_false(densenet):
    assert densenet.is_vit() is False


def test_densenet_num_classes(densenet):
    assert densenet.num_classes == 14


def test_densenet_dropout():
    m = build_model("densenet121", num_classes=14, pretrained=False, dropout=0.5)
    # Head should contain a Dropout layer
    head_types = [type(layer).__name__ for layer in m.head]
    assert "Dropout" in head_types


# ── CXRClassifier — ViT-Base/16 ───────────────────────────────────────────────

def test_vit_output_shape(vit, dummy_batch):
    vit.eval()
    with torch.no_grad():
        out = vit(dummy_batch)
    assert out.shape == (2, 14)


def test_vit_is_vit_true(vit):
    assert vit.is_vit() is True


def test_vit_num_classes(vit):
    assert vit.num_classes == 14


# ── Layer-wise LR decay ───────────────────────────────────────────────────────

def test_densenet_param_groups_lr_ordering(densenet):
    """Outer groups must have higher or equal lr than inner groups."""
    groups = get_layerwise_param_groups(densenet, base_lr=1e-4, decay_factor=0.9)
    lrs = [g["lr"] for g in groups]
    assert lrs == sorted(lrs, reverse=True), "LR should decrease outer→inner"


def test_vit_param_groups_lr_ordering(vit):
    groups = get_layerwise_param_groups(vit, base_lr=1e-4, decay_factor=0.9)
    lrs = [g["lr"] for g in groups]
    assert lrs == sorted(lrs, reverse=True)


def test_param_groups_cover_all_params(densenet):
    """Every model parameter must appear in exactly one param group."""
    groups = get_layerwise_param_groups(densenet, base_lr=1e-4)
    grouped_ids = set()
    for g in groups:
        for p in g["params"]:
            grouped_ids.add(id(p))
    all_ids = {id(p) for p in densenet.parameters()}
    assert all_ids == grouped_ids, "Some parameters are missing from param groups"


def test_param_groups_no_duplicates(densenet):
    """No parameter should appear in more than one group."""
    groups = get_layerwise_param_groups(densenet, base_lr=1e-4)
    seen = []
    for g in groups:
        for p in g["params"]:
            assert id(p) not in seen, "Duplicate parameter in param groups"
            seen.append(id(p))


def test_param_groups_weight_decay(densenet):
    groups = get_layerwise_param_groups(densenet, base_lr=1e-4, weight_decay=1e-5)
    for g in groups:
        assert g["weight_decay"] == pytest.approx(1e-5)
