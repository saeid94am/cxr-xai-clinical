"""
Tests for src/data — dataset classes and transforms.

All tests run without real NIH-14 images: synthetic dummy data is written
to a temporary directory so CI requires no external downloads.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    CHEXPERT_EVAL_CLASSES,
    NIH14_CLASSES,
    CheXpertValDataset,
    NIH14Dataset,
)
from src.data.transforms import train_transforms, val_transforms

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_dummy_png(path: Path, size: int = 64) -> None:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture()
def nih14_tmp(tmp_path: Path):
    """Create a minimal NIH-14-like directory with 10 dummy images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    n = 10
    filenames = [f"img_{i:04d}.png" for i in range(n)]
    labels = [NIH14_CLASSES[i % len(NIH14_CLASSES)] for i in range(n)]

    for fname in filenames:
        _make_dummy_png(img_dir / fname)

    # Data_Entry CSV
    csv_path = tmp_path / "Data_Entry_2017.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Image Index",
                "Finding Labels",
                "Follow-up #",
                "Patient ID",
                "Patient Age",
                "Patient Gender",
                "View Position",
                "OriginalImageWidth",
                "OriginalImageHeight",
                "OriginalImagePixelSpacing_x",
                "OriginalImagePixelSpacing_y",
                "Unnamed: 11",
            ]
        )
        for fname, label in zip(filenames, labels):
            writer.writerow([fname, label] + [""] * 10)

    # Split file (all images in train)
    split_path = tmp_path / "train_val_list.txt"
    split_path.write_text("\n".join(filenames))

    return {"img_dir": str(img_dir), "csv": str(csv_path), "split": str(split_path), "n": n}


@pytest.fixture()
def chexpert_tmp(tmp_path: Path):
    """Create a minimal CheXpert-like val CSV with 5 dummy images."""
    img_dir = tmp_path / "CheXpert-v1.0-small" / "valid"
    img_dir.mkdir(parents=True)

    n = 5
    paths, rows = [], []
    for i in range(n):
        rel = f"CheXpert-v1.0-small/valid/patient{i:05d}/study1/view1.jpg"
        abs_path = tmp_path / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        _make_dummy_png(abs_path)
        paths.append(rel)
        row = {"Path": rel}
        for cls in CHEXPERT_EVAL_CLASSES:
            row[cls] = float(i % 2)  # alternating 0 / 1
        rows.append(row)

    import pandas as pd

    csv_path = tmp_path / "valid.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {"root": str(tmp_path), "csv": str(csv_path), "n": n}


# ── Transform tests ───────────────────────────────────────────────────────────


def test_train_transforms_output_shape():
    t = train_transforms(224)
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_output_shape():
    t = val_transforms(224)
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_deterministic():
    """Val transform must produce identical output for the same input."""
    t = val_transforms(224)
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))
    assert torch.equal(t(img), t(img))


# ── NIH14Dataset tests ────────────────────────────────────────────────────────


def test_nih14_dataset_length(nih14_tmp):
    ds = NIH14Dataset(
        nih14_tmp["img_dir"],
        nih14_tmp["csv"],
        nih14_tmp["split"],
        transform=val_transforms(224),
    )
    assert len(ds) == nih14_tmp["n"]


def test_nih14_dataset_item_shapes(nih14_tmp):
    ds = NIH14Dataset(
        nih14_tmp["img_dir"],
        nih14_tmp["csv"],
        nih14_tmp["split"],
        transform=val_transforms(224),
    )
    img, label, path = ds[0]
    assert img.shape == (3, 224, 224)
    assert label.shape == (len(NIH14_CLASSES),)
    assert label.dtype == torch.float32
    assert isinstance(path, str)


def test_nih14_dataset_labels_binary(nih14_tmp):
    ds = NIH14Dataset(
        nih14_tmp["img_dir"],
        nih14_tmp["csv"],
        nih14_tmp["split"],
        transform=val_transforms(224),
    )
    for _, label, _ in ds:
        assert set(label.numpy().tolist()).issubset({0.0, 1.0})


def test_nih14_dataset_subset(nih14_tmp):
    ds = NIH14Dataset(
        nih14_tmp["img_dir"],
        nih14_tmp["csv"],
        nih14_tmp["split"],
        transform=val_transforms(224),
        subset_n=5,
    )
    assert len(ds) == 5


def test_nih14_pos_weights_shape(nih14_tmp):
    ds = NIH14Dataset(
        nih14_tmp["img_dir"],
        nih14_tmp["csv"],
        nih14_tmp["split"],
        transform=val_transforms(224),
    )
    pw = ds.compute_pos_weights()
    assert pw.shape == (len(NIH14_CLASSES),)
    assert (pw > 0).all()


# ── CheXpertValDataset tests ──────────────────────────────────────────────────


def test_chexpert_dataset_length(chexpert_tmp):
    ds = CheXpertValDataset(
        chexpert_tmp["root"],
        chexpert_tmp["csv"],
        transform=val_transforms(224),
    )
    assert len(ds) == chexpert_tmp["n"]


def test_chexpert_dataset_item_shapes(chexpert_tmp):
    ds = CheXpertValDataset(
        chexpert_tmp["root"],
        chexpert_tmp["csv"],
        transform=val_transforms(224),
    )
    img, label, path = ds[0]
    assert img.shape == (3, 224, 224)
    assert label.shape == (len(CHEXPERT_EVAL_CLASSES),)


def test_chexpert_no_uncertain_labels(chexpert_tmp):
    """Uncertain labels (-1) must be mapped to 0."""
    import pandas as pd

    # Inject -1 values into the CSV
    df = pd.read_csv(chexpert_tmp["csv"])
    df[CHEXPERT_EVAL_CLASSES[0]] = -1
    df.to_csv(chexpert_tmp["csv"], index=False)

    ds = CheXpertValDataset(
        chexpert_tmp["root"],
        chexpert_tmp["csv"],
        transform=val_transforms(224),
    )
    for _, label, _ in ds:
        assert (label >= 0).all(), "Uncertain label (-1) was not mapped to 0"
