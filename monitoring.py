"""Monitoring module with PSI, AUC and recall-by-decile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def psi_flag(psi: float) -> str:
    """Return alert label according to PSI thresholds."""
    if psi < 0.10:
        return "OK"
    if psi < 0.25:
        return "WARN"
    return "ALERT"


def calculate_psi(expected_array: np.ndarray, actual_array: np.ndarray, buckets: int = 10) -> float:
    """Compute population stability index using expected distribution breakpoints."""
    breakpoints = np.percentile(expected_array, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_perc = np.histogram(expected_array, bins=breakpoints)[0] / max(len(expected_array), 1)
    actual_perc = np.histogram(actual_array, bins=breakpoints)[0] / max(len(actual_array), 1)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    return float(np.sum(psi_values))


def compute_recall_by_decile(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute cumulative recall by decile (1 = highest score)."""
    df = pd.DataFrame({"score": np.asarray(scores), "target": np.asarray(y_true)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Ranking-based split avoids qcut failures when many repeated scores exist.
    df["decil"] = pd.qcut(df.index + 1, q=n_deciles, labels=range(1, n_deciles + 1))

    total_positives = max(int(df["target"].sum()), 1)
    grouped = (
        df.groupby("decil", observed=False)["target"]
        .sum()
        .rename("positives")
        .reset_index()
        .sort_values("decil")
    )
    grouped["positives_acum"] = grouped["positives"].cumsum()
    grouped["recall_acumulado"] = grouped["positives_acum"] / total_positives
    return grouped[["decil", "recall_acumulado"]]


def run_monitoring(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    target_col: str = "target",
    output_dir: str | Path = "data/monitoring",
    mlflow_active: bool = False,
) -> dict[str, Any]:
    """Compute monitoring metrics and persist outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psi_score = calculate_psi(np.asarray(train_scores), np.asarray(val_scores), buckets=10)
    psi_status = psi_flag(psi_score)

    val_auc = float("nan")
    if target_col in df_val.columns and df_val[target_col].nunique() > 1:
        val_auc = float(roc_auc_score(df_val[target_col], val_scores))

    recall_df = compute_recall_by_decile(df_val[target_col], val_scores)
    recall_df.to_csv(output_dir / "recall_by_decile.csv", index=False)

    summary: dict[str, Any] = {
        "psi_score": float(psi_score),
        "psi_status": psi_status,
        "val_auc": val_auc,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
    }

    with (output_dir / "monitoring_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if mlflow_active and mlflow is not None:
        mlflow.log_metric("monitoring.psi", psi_score)
        if not np.isnan(val_auc):
            mlflow.log_metric("monitoring.val_auc", val_auc)

    return summary
