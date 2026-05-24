"""
Pointing Game accuracy using NIH-14 bounding box annotations.

Definition: a hit is scored when the pixel with the highest saliency value
in the heatmap falls inside the ground-truth bounding box for that pathology.
Accuracy = hits / total annotated images, reported per pathology and overall.

Reference: Zhang et al. (2016) "Top-down neural attention by excitation backprop"
Ground truth: BBox_List_2017.csv from the NIH ChestX-ray14 release (984 boxes,
8 pathologies).

BBox_List_2017.csv columns:
    Image Index, Finding Label, Bbox [x, y, w, h]
    (x, y) is top-left corner; w, h are width and height in pixels.
    Coordinates are for the original 1024×1024 images — scaled here to
    match the model input resolution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_bbox_df(bbox_csv: str) -> pd.DataFrame:
    """Load and normalise the NIH-14 bounding box CSV.

    Returns a DataFrame with columns:
        filename, label, x, y, w, h  (all pixel coords at 1024×1024)
    """
    df = pd.read_csv(bbox_csv)
    df.columns = [c.strip() for c in df.columns]

    # Rename to consistent internal names
    rename = {
        "Image Index": "filename",
        "Finding Label": "label",
        "Bbox [x": "x",
        "y": "y",
        "w": "w",
        "h]": "h",
    }
    df = df.rename(columns=rename)

    # Some CSV versions have slightly different column names; handle both
    if "x" not in df.columns:
        # Try splitting a combined bbox column
        bbox_cols = [c for c in df.columns if "Bbox" in c or "bbox" in c]
        if bbox_cols:
            parts = df[bbox_cols[0]].str.extract(
                r"(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)"
            )
            df["x"], df["y"], df["w"], df["h"] = (
                parts[0].astype(float),
                parts[1].astype(float),
                parts[2].astype(float),
                parts[3].astype(float),
            )

    df[["x", "y", "w", "h"]] = df[["x", "y", "w", "h"]].astype(float)
    return df[["filename", "label", "x", "y", "w", "h"]]


def _scale_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    orig_size: int = 1024,
    target_size: int = 224,
) -> Tuple[int, int, int, int]:
    """Scale bbox coords from orig_size to target_size."""
    scale = target_size / orig_size
    return (
        int(x * scale),
        int(y * scale),
        int(w * scale),
        int(h * scale),
    )


def pointing_game_hit(
    heatmap: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tolerance_px: int = 0,
) -> bool:
    """Return True if the argmax pixel of heatmap falls inside bbox.

    Args:
        heatmap:      (H, W) float array — saliency map.
        bbox:         (x, y, w, h) bounding box in heatmap pixel coords.
        tolerance_px: Expand bbox by this many pixels on each side.

    Returns:
        True = hit, False = miss.
    """
    H, W = heatmap.shape
    x, y, w, h = bbox

    # Find argmax pixel
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, W)

    x1 = max(0, x - tolerance_px)
    y1 = max(0, y - tolerance_px)
    x2 = min(W, x + w + tolerance_px)
    y2 = min(H, y + h + tolerance_px)

    return (x1 <= px < x2) and (y1 <= py < y2)


def compute_pointing_game(
    heatmaps: Dict[str, Dict[str, np.ndarray]],
    bbox_csv: str,
    img_size: int = 224,
    tolerance_px: int = 0,
) -> pd.DataFrame:
    """Compute pointing game accuracy across all annotated images.

    Args:
        heatmaps:    Nested dict: heatmaps[filename][label] = (H, W) array.
        bbox_csv:    Path to BBox_List_2017.csv.
        img_size:    Resolution of the heatmaps (default 224).
        tolerance_px: Pixel tolerance around bbox edge (0 = exact).

    Returns:
        DataFrame with columns [label, hits, total, accuracy].
    """
    bbox_df = load_bbox_df(bbox_csv)

    results: Dict[str, Dict[str, int]] = {}  # label → {hits, total}

    for _, row in bbox_df.iterrows():
        fname: str = row["filename"]
        label: str = row["label"]

        if fname not in heatmaps or label not in heatmaps[fname]:
            continue

        heatmap = heatmaps[fname][label]
        bbox = _scale_bbox(
            row["x"], row["y"], row["w"], row["h"], orig_size=1024, target_size=img_size
        )
        hit = pointing_game_hit(heatmap, bbox, tolerance_px)

        if label not in results:
            results[label] = {"hits": 0, "total": 0}
        results[label]["hits"] += int(hit)
        results[label]["total"] += 1

    rows = []
    for label, counts in results.items():
        acc = counts["hits"] / counts["total"] if counts["total"] > 0 else float("nan")
        rows.append(
            {"label": label, "hits": counts["hits"], "total": counts["total"], "accuracy": acc}
        )

    if not rows:
        return pd.DataFrame(columns=["label", "hits", "total", "accuracy"])

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    if not df.empty:
        overall_acc = df["hits"].sum() / df["total"].sum()
        overall_row = pd.DataFrame(
            [
                {
                    "label": "OVERALL",
                    "hits": df["hits"].sum(),
                    "total": df["total"].sum(),
                    "accuracy": overall_acc,
                }
            ]
        )
        df = pd.concat([df, overall_row], ignore_index=True)

    return df
