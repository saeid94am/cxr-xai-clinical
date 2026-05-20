"""
Assemble and save the paper's Table 2: XAI comparison across all methods and metrics.

Expected columns (matches Phase 2 plan):
  XAI Method | Backbone | Pointing Game ↑ | Deletion AUC ↓ |
  Insertion AUC ↑ | Spearman ρ ↑ | ROAD ↑ | Sanity Check
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_summary_row(
    method: str,
    backbone: str,
    pointing_game: float | None = None,
    deletion_auc: float | None = None,
    insertion_auc: float | None = None,
    spearman_rho: float | None = None,
    road: float | None = None,
    sanity_pass: bool | None = None,
) -> Dict[str, Any]:
    """Build one row dict for the XAI comparison table."""
    return {
        "XAI Method":      method,
        "Backbone":        backbone,
        "Pointing Game ↑": round(pointing_game, 4)  if pointing_game  is not None else None,
        "Deletion AUC ↓":  round(deletion_auc, 4)   if deletion_auc   is not None else None,
        "Insertion AUC ↑": round(insertion_auc, 4)  if insertion_auc  is not None else None,
        "Spearman ρ ↑":    round(spearman_rho, 4)   if spearman_rho   is not None else None,
        "ROAD ↑":          round(road, 4)            if road           is not None else None,
        "Sanity Check":    ("Pass" if sanity_pass else "Fail") if sanity_pass is not None else None,
    }


def save_summary_table(rows: list[Dict[str, Any]], output_path: str) -> pd.DataFrame:
    """Write the XAI comparison table to CSV and return it as a DataFrame.

    Args:
        rows:        List of dicts from build_summary_row().
        output_path: Path to write the CSV (e.g. results/metrics/xai_comparison_table.csv).

    Returns:
        DataFrame of the full table.
    """
    df = pd.DataFrame(rows)

    # Canonical row order matching the paper table
    order = ["Grad-CAM++", "HiResCAM", "Integrated Gradients", "Attention Rollout"]
    df["_order"] = df["XAI Method"].apply(
        lambda m: order.index(m) if m in order else len(order)
    )
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[summary] XAI comparison table saved → {output_path}")
    return df
