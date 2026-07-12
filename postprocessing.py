"""Postprocessing module for TLV score and replica file generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Business quantiles provided in the assignment and must remain unchanged.
DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]


def get_groups(scores: np.ndarray, df_post: pd.DataFrame) -> pd.DataFrame:
    """Calculate TLV score and assign execution group (1-10)."""
    df_post = df_post.copy()
    df_post["prob"] = np.asarray(scores)

    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1",
        0.066,
        np.where(
            df_post["grp_campecs06m"] == "G2",
            0.028,
            np.where(
                df_post["grp_campecs06m"] == "G3",
                0.022,
                np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004),
            ),
        ),
    )

    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"].fillna(0) + 1)
        * df_post["prob_frescura"]
    )

    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"],
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        duplicates="drop",
    )
    return df_post


def run_postprocessing(
    scores: np.ndarray,
    df_post: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Wrapper of get_groups with optional CSV persistence."""
    result = get_groups(scores, df_post)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def _build_replica_dataframe(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
) -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")

    replica = pd.DataFrame(
        {
            "codmes": partition,
            "tipdoc": df_post.get("tip_doc", "1"),
            "coddoc": df_post.get("key_value", df_post.get("codunicocli", "NA")),
            "puntuacion": df_post["puntuacion_tlv"],
            "modelo": table,
            "fec_replica": today,
            "grupo_ejec": df_post["grupo_ejec_tlv"],
            "score": df_post["prob"],
            "orden": np.arange(1, len(df_post) + 1),
            "variable1": df_post.get("monto", np.nan),
            "variable2": df_post.get("prob_value_contact", np.nan),
            "variable3": df_post.get("grp_campecs06m", "NA"),
        }
    )
    return replica


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
    dir_s3: str | Path = "data/replica/s3",
    dir_athena: str | Path = "data/replica/athena",
    dir_onpremise: str | Path = "data/replica/onpremise",
) -> list[str]:
    """Generate the same pipe-delimited replica for three destinations."""
    replica = _build_replica_dataframe(df_post, table=table, partition=partition)

    output_dirs = [Path(dir_s3), Path(dir_athena), Path(dir_onpremise)]
    output_paths: list[str] = []
    for base_dir in output_dirs:
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{table}_{partition}.txt"
        replica.to_csv(path, sep="|", index=False)
        output_paths.append(str(path))

    return output_paths
