"""
Dataset classes for NIH ChestX-ray14, CheXpert (val set), and RSNA Pneumonia.

NIH-14 canonical label order matches configs/train.yaml `nih14_classes`.
All datasets return (image_tensor, label_tensor, image_path) for traceability.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── NIH ChestX-ray14 label columns (canonical order) ─────────────────────────
NIH14_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# ── CheXpert 5-class eval subset ──────────────────────────────────────────────
CHEXPERT_EVAL_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class NIH14Dataset(Dataset):
    """NIH ChestX-ray14 multi-label classification dataset.

    Args:
        image_dir:   Directory containing flat PNG images.
        labels_csv:  Path to Data_Entry_2017.csv.
        split_file:  Path to train_val_list.txt or test_list.txt (one filename per line).
        transform:   Image transform pipeline.
        subset_n:    If set, randomly sample this many images (debug_mode).
        seed:        Random seed for reproducible subset sampling.
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        split_file: str,
        transform: Optional[Callable] = None,
        subset_n: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        with open(split_file) as f:
            split_files = {line.strip() for line in f if line.strip()}
        df = df[df["Image Index"].isin(split_files)].reset_index(drop=True)

        if subset_n is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=min(subset_n, len(df)), replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        self.filenames = df["Image Index"].tolist()
        # Build binary label matrix
        finding_col = df["Finding Labels"].str.split("|")
        labels = np.zeros((len(df), len(NIH14_CLASSES)), dtype=np.float32)
        for i, findings in enumerate(finding_col):
            for f in findings:
                if f in NIH14_CLASSES:
                    labels[i, NIH14_CLASSES.index(f)] = 1.0
        self.labels = labels

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]
        img_path = self.image_dir / fname
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        return img, label, str(img_path)

    @property
    def classes(self) -> List[str]:
        return NIH14_CLASSES

    def compute_pos_weights(self) -> torch.Tensor:
        """Per-class positive weights for BCEWithLogitsLoss (handles imbalance)."""
        pos = self.labels.sum(axis=0)
        neg = len(self.labels) - pos
        # Avoid division by zero; clamp minimum pos count to 1
        pos = np.maximum(pos, 1)
        return torch.from_numpy(neg / pos)


class CheXpertValDataset(Dataset):
    """CheXpert validation set — 5-pathology cross-dataset evaluation only.

    Uncertainty labels (-1) are treated as negative (conservative policy)
    since we are using NIH-14-trained models for evaluation, not retraining.

    Args:
        root:      CheXpert root directory.
        valid_csv: Path to valid.csv.
        transform: Image transform pipeline.
    """

    def __init__(
        self,
        root: str,
        valid_csv: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        df = pd.read_csv(valid_csv)
        self.paths = df["Path"].tolist()

        labels = np.zeros((len(df), len(CHEXPERT_EVAL_CLASSES)), dtype=np.float32)
        for i, cls in enumerate(CHEXPERT_EVAL_CLASSES):
            col = df[cls].fillna(0).values
            # -1 (uncertain) → 0 (negative)
            labels[:, i] = np.where(col == 1, 1.0, 0.0)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.root / self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        return img, label, str(img_path)

    @property
    def classes(self) -> List[str]:
        return CHEXPERT_EVAL_CLASSES


class RSNAPneumoniaDataset(Dataset):
    """RSNA Pneumonia Detection dataset — domain shift XAI experiment.

    Uses DICOM images; falls back to PNG if pre-converted.

    Args:
        image_dir:   Directory containing patient sub-dirs with DICOM files,
                     or a flat directory of PNG files.
        labels_csv:  Path to stage_2_train_labels.csv.
        transform:   Image transform pipeline.
        use_png:     True if images were pre-converted to PNG.
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform: Optional[Callable] = None,
        use_png: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.use_png = use_png

        df = pd.read_csv(labels_csv)
        # Keep one row per patient (stage_2 has multiple boxes per patient)
        df = df.drop_duplicates(subset="patientId").reset_index(drop=True)
        self.patient_ids = df["patientId"].tolist()
        self.labels = df["Target"].astype(np.float32).values.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pid = self.patient_ids[idx]
        if self.use_png:
            img_path = self.image_dir / f"{pid}.png"
            img = Image.open(img_path).convert("RGB")
        else:
            try:
                import pydicom
            except ImportError as e:
                raise ImportError(
                    "pydicom is required for DICOM loading. pip install pydicom"
                ) from e
            img_path = self.image_dir / f"{pid}.dcm"
            dcm = pydicom.dcmread(str(img_path))
            arr = dcm.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        return img, label, str(img_path)

    @property
    def classes(self) -> List[str]:
        return ["Pneumonia"]


# ── DataLoader factory ────────────────────────────────────────────────────────


def build_nih14_loaders(
    image_dir: str,
    labels_csv: str,
    train_val_list: str,
    test_list: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    debug_n: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for NIH-14."""
    from .transforms import train_transforms, val_transforms

    train_ds = NIH14Dataset(
        image_dir,
        labels_csv,
        train_val_list,
        transform=train_transforms(img_size),
        subset_n=debug_n,
        seed=seed,
    )
    # Use the same split file for val — NIH-14 provides a single train+val file;
    # we do an 80/20 random split internally here.
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split].tolist(), idx[split:].tolist()

    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_ds_full = NIH14Dataset(
        image_dir,
        labels_csv,
        train_val_list,
        transform=val_transforms(img_size),
        subset_n=debug_n,
        seed=seed,
    )
    val_subset = torch.utils.data.Subset(val_ds_full, val_idx)

    test_ds = NIH14Dataset(
        image_dir,
        labels_csv,
        test_list,
        transform=val_transforms(img_size),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader, test_loader
