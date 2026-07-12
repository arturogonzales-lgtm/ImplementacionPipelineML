"""Main orchestrator for ML E2E pipeline.

Run example:
python main.py --input_dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monitoring import run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import run_preprocessing
from training import train_and_log


ID_COL_CANDIDATES = [
    "partition",
    "key_value",
    "codunicocli",
    "tip_doc",
    "fch_creacion",
    "p_fecinformacion",
    "_source_file",
]
TARGET_COL = "target"


def _feature_frame(df):
    drop_cols = [c for c in ID_COL_CANDIDATES + [TARGET_COL] if c in df.columns]
    return df.drop(columns=drop_cols)


def _need_retraining(monitor_summary: dict, post_df) -> bool:
    psi_status = monitor_summary.get("psi_status", "OK")
    groups_ok = "grupo_ejec_tlv" in post_df.columns and post_df["grupo_ejec_tlv"].notna().sum() > 0
    return psi_status == "ALERT" or not groups_ok


def run_pipeline(args: argparse.Namespace) -> None:
    print("[1/5] Preprocessing...")
    df_train, _, df_val, meta = run_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.processed_dir,
        nan_threshold=args.nan_threshold,
        validation_partition=args.validation_partition,
    )

    print("[2/5] Training with HPO...")
    run_id, model, train_metrics = train_and_log(
        train_path=Path(args.processed_dir) / "df_train.csv",
        test_path=Path(args.processed_dir) / "df_test.csv",
        val_path=Path(args.processed_dir) / "df_val.csv",
        model_dir=args.model_dir,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
    )

    print("[3/5] Monitoring...")
    X_train = _feature_frame(df_train)
    x_val = _feature_frame(df_val)

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(x_val)[:, 1]

    monitor_summary = run_monitoring(
        df_train=df_train,
        df_val=df_val,
        train_scores=train_scores,
        val_scores=val_scores,
        target_col=TARGET_COL,
        output_dir=args.monitoring_dir,
        mlflow_active=True,
    )

    print("[4/5] Postprocessing + replica...")
    df_post = run_postprocessing(
        scores=val_scores,
        df_post=df_val,
        output_path=args.post_path,
    )

    partition = args.replica_partition or str(meta.get("validation_partition") or "unknown")
    replica_paths = save_replica(
        df_post,
        table=args.replica_table,
        partition=partition,
    )

    if _need_retraining(monitor_summary, df_post):
        print("[5/5] Monitoring alert detected (PSI/group matrix). Automatic retraining...")
        run_id, model, train_metrics = train_and_log(
            train_path=Path(args.processed_dir) / "df_train.csv",
            test_path=Path(args.processed_dir) / "df_test.csv",
            val_path=Path(args.processed_dir) / "df_val.csv",
            model_dir=args.model_dir,
            n_trials=max(args.n_trials, 40),
            experiment_name=args.experiment_name,
        )
    else:
        print("[5/5] Pipeline finished without retraining alerts.")

    print("----- SUMMARY -----")
    print(f"Run ID: {run_id}")
    print(f"Train/Test/Val AUC: {train_metrics['train_auc']:.4f} / {train_metrics['test_auc']:.4f} / {train_metrics['val_auc']:.4f}")
    print(f"PSI: {monitor_summary['psi_score']:.4f} ({monitor_summary['psi_status']})")
    print(f"Postprocessed output: {args.post_path}")
    print("Replica outputs:")
    for p in replica_paths:
        print(f" - {p}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CU Venta ML E2E pipeline")
    parser.add_argument("--input_dir", default="data/raw", help="Directory with raw CSV files")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed output directory")
    parser.add_argument("--model_dir", default="data/models", help="Directory to save trained model")
    parser.add_argument("--monitoring_dir", default="data/monitoring", help="Monitoring artifacts directory")
    parser.add_argument("--post_path", default="data/postprocessed/output_tlv.csv", help="Postprocessed output CSV path")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--nan_threshold", type=float, default=80.0, help="Drop columns above this NaN percent")
    parser.add_argument("--validation_partition", default=None, help="Optional explicit validation partition")
    parser.add_argument("--experiment_name", default="cu_venta_e2e", help="MLflow experiment name")
    parser.add_argument("--replica_table", default="EC_OMNICANAL", help="Replica output model/table name")
    parser.add_argument("--replica_partition", default=None, help="Replica partition (e.g. 202412)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run_pipeline(parser.parse_args())
