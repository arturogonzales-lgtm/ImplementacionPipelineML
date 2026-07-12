"""Preprocessing module for CU Venta pipeline.

This module reads all raw CSV files, applies cleaning/imputation/encoding,
creates train/test/validation splits, and persists processed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Business defaults inspired by the class notebooks.
NAN_THRESHOLD = 80.0
TEST_SIZE = 0.30
RANDOM_STATE = 123
TARGET_COL = "target"

ZERO_IMPUTE_COLS = [
    "flg_saltothip12m",
    "flg_saltotppe12m",
    "prm_pctsaltototrent12m",
    "prm_pctsaltotcaja12m",
    "ant_saltot24m",
    "ant_saltot12m",
    "min_difsaltottcr12m",
    "num_incrsaldispefe06m",
    "max_difent12m",
    "num_dismsalppecons06m",
    "beta_pctusotcr12m",
    "prm_pctusosaltottcr03m",
    "dsv_saltotppe03m",
    "prm_diasatrrdpn12m",
    "dsv_numentrdlintcr03m",
    "rat_disefepnm01",
    "prm_diasatrrdpn06m",
    "pct_usotcrm01",
    "dsv_numentrdlintcr06m",
    "beta_saltotppe12m",
    "prm_entrd03m",
    "ctd_entrdm01",
    "beta_saltotppe06m",
    "prm_diasatrrd03m",
    "prm_saltotrdpj03m",
    "prm_saltotrdpj12m",
    "dsv_diasatrrdpj12m",
    "max_pctsalimpago12m",
    "prm_diasatrrdpj03m",
    "ctd_campecstlv06m",
    "max_camptot06m",
    "min_camptot06m",
    "frc_camptot06m",
    "rec_camptot06m",
    "ctd_camptot06m",
    "prm_camptot06m",
    "max_campecs06m",
    "min_campecs06m",
    "frc_campecs06m",
    "rec_campecs06m",
    "ctd_campecs06m",
    "prm_campecs06m",
    "seg_un",
]

CATEGORICAL_FILL_VALUES = {
    "ubigeo_buro": "Otros",
    "grp_riesgociiu": "grupo_0",
    "grp_camptot06m": "Otro",
    "grp_campecs06m": "Otro",
    "region": "Otro",
}

NON_ENCODE_COLS = {
    "_source_file",
    "partition",
    "key_value",
    "codunicocli",
    "fch_creacion",
    "p_fecinformacion",
}


def _infer_validation_partition(df: pd.DataFrame) -> str | None:
    if "partition" not in df.columns:
        return None
    partitions = sorted(df["partition"].dropna().astype(str).unique())
    if not partitions:
        return None
    return partitions[-1]


def _business_recoding(df: pd.DataFrame) -> pd.DataFrame:
    if "seg_un" in df.columns:
        df["seg_un"] = np.where(df["seg_un"].isin([0, 3]), 0, df["seg_un"])

    if "grp_riesgociiu" in df.columns:
        group_map = ["grupo_2", "grupo_3", "grupo_9", "grupo_8", "grupo_1"]
        df["grp_riesgociiu"] = np.where(
            df["grp_riesgociiu"].isin(group_map),
            "grupo_11",
            df["grp_riesgociiu"],
        )
    return df


def _safe_json(data: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(data)


def _load_input_data(input_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["_source_file"] = csv_path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return df, csv_paths


def _apply_imputations(df: pd.DataFrame) -> pd.DataFrame:
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "edad" in df.columns:
        df["edad"] = df["edad"].fillna(df["edad"].median())

    for col, value in CATEGORICAL_FILL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df


def _encode_object_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    encoders: dict[str, list[str]] = {}
    object_cols = [c for c in df.columns if str(df[c].dtype) in {"object", "string"} and c not in NON_ENCODE_COLS]
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le.classes_.tolist()
    return df, encoders


def _split_dataset(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    validation_partition: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    if validation_partition is None:
        validation_partition = _infer_validation_partition(df)

    if validation_partition and "partition" in df.columns:
        val_mask = df["partition"].astype(str) == str(validation_partition)
        if val_mask.any():
            df_val = df[val_mask].copy()
            df_main = df[~val_mask].copy()
        else:
            inferred = _infer_validation_partition(df)
            if inferred is not None and inferred != validation_partition:
                validation_partition = inferred
                val_mask = df["partition"].astype(str) == str(validation_partition)
                df_val = df[val_mask].copy()
                df_main = df[~val_mask].copy()
            else:
                df_main, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
                validation_partition = None
    else:
        df_main, df_val = train_test_split(df, test_size=test_size, random_state=random_state)

    stratify_col = df_main[TARGET_COL] if TARGET_COL in df_main.columns else None
    df_train, df_test = train_test_split(
        df_main,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    return df_train, df_test, df_val, validation_partition


def run_preprocessing(
    input_dir: str | Path,
    output_dir: str | Path = "data/processed",
    nan_threshold: float = NAN_THRESHOLD,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    validation_partition: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run full preprocessing and save train/test/validation CSVs.

    Returns:
        df_train, df_test, df_val, metadata
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, csv_paths = _load_input_data(input_dir)
    df = df.replace(["", "null", "None"], np.nan)

    cols_drop = [c for c in df.columns if df[c].isna().mean() * 100 > nan_threshold]
    df = df.drop(columns=cols_drop)

    df = _apply_imputations(df)
    df = _business_recoding(df)
    df, encoders = _encode_object_columns(df)
    df_train, df_test, df_val, validation_partition = _split_dataset(
        df=df,
        test_size=test_size,
        random_state=random_state,
        validation_partition=validation_partition,
    )

    train_path = output_dir / "df_train.csv"
    test_path = output_dir / "df_test.csv"
    val_path = output_dir / "df_val.csv"
    meta_path = output_dir / "metadata.json"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_val.to_csv(val_path, index=False)

    metadata: dict[str, Any] = {
        "input_files": [p.name for p in csv_paths],
        "rows_total": int(len(df)),
        "rows_train": int(len(df_train)),
        "rows_test": int(len(df_test)),
        "rows_val": int(len(df_val)),
        "dropped_columns": cols_drop,
        "encoders": encoders,
        "validation_partition": validation_partition,
        "target_col": TARGET_COL,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "val_path": str(val_path),
    }

    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(_safe_json(metadata), fp, indent=2)

    return df_train, df_test, df_val, metadata
