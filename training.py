"""Training module with Optuna HPO, stability checks, and MLflow logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import mlflow
    import mlflow.xgboost
except Exception:  # pragma: no cover
    mlflow = None

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required for training. Install with: pip install xgboost") from exc

ID_COLS = [
    "partition",
    "key_value",
    "codunicocli",
    "tip_doc",
    "fch_creacion",
    "p_fecinformacion",
    "_source_file",
]
TARGET_COL = "target"


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COL}' was not found")
    return df.drop(columns=drop_cols), df[TARGET_COL]


def _stability_penalty(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int,
) -> tuple[float, float]:
    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def train_and_log(
    train_path: str | Path,
    test_path: str | Path,
    val_path: str | Path,
    model_dir: str | Path = "data/models",
    n_trials: int = 30,
    experiment_name: str = "cu_venta_e2e",
    stability_weight: float = 0.15,
    max_allowed_decay: float = 10.0,
    cv_splits: int = 5,
    random_state: int = 123,
) -> tuple[str, xgb.XGBClassifier, dict[str, Any]]:
    """Train with Optuna and return run_id, model and metrics metadata."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    val_path = Path(val_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)

    X_train, y_train = _xy(df_train)
    X_test, y_test = _xy(df_test)
    x_val, y_val = _xy(df_val)

    # Align columns defensively in case preprocessing creates mismatch.
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
        x_val[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    for col in set(x_val.columns) - set(X_train.columns):
        X_train[col] = 0

    X_test = X_test[X_train.columns]
    x_val = x_val[X_train.columns]

    def evaluate_params(params: dict[str, Any]) -> float:

        _, cv_std = _stability_penalty(
            params=params,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            random_state=random_state,
        )

        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        decay = ((train_auc - test_auc) / max(train_auc, 1e-12)) * 100

        decay_penalty = 1.0 if decay > max_allowed_decay else 0.0
        # Goal-backward objective: performance + stability + overfit control.
        return float(test_auc - (stability_weight * cv_std) - decay_penalty)

    if optuna is not None:
        def objective(trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 120, 700),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
            return evaluate_params(params)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = {**study.best_params}
    else:
        rng = np.random.default_rng(seed=random_state)
        best_score = -np.inf
        best_params: dict[str, Any] = {}
        for _ in range(n_trials):
            params = {
                "n_estimators": int(rng.integers(120, 701)),
                "max_depth": int(rng.integers(3, 11)),
                "learning_rate": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2)))),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": int(rng.integers(1, 13)),
                "gamma": float(rng.uniform(0.0, 5.0)),
                "reg_alpha": float(rng.uniform(0.0, 5.0)),
                "reg_lambda": float(rng.uniform(0.0, 5.0)),
            }
            score = evaluate_params(params)
            if score > best_score:
                best_score = score
                best_params = params
    model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
    decay_pct = ((train_auc - test_auc) / max(train_auc, 1e-12)) * 100
    cv_mean, cv_std = _stability_penalty(best_params, X_train, y_train, cv_splits, random_state)

    run_id = "local-run"
    if mlflow is not None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(best_params)
            mlflow.log_metric("train.auc", train_auc)
            mlflow.log_metric("test.auc", test_auc)
            mlflow.log_metric("val.auc", val_auc)
            mlflow.log_metric("test.decay_pct", decay_pct)
            mlflow.log_metric("cv.mean_auc", cv_mean)
            mlflow.log_metric("cv.std_auc", cv_std)
            mlflow.xgboost.log_model(model, "model")

    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"

    joblib.dump(model, model_path)

    metrics = {
        "run_id": run_id,
        "hpo_engine": "optuna" if optuna is not None else "random_search",
        "best_params": best_params,
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "val_auc": float(val_auc),
        "decay_pct": float(decay_pct),
        "cv_mean_auc": float(cv_mean),
        "cv_std_auc": float(cv_std),
        "model_path": str(model_path),
    }

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return run_id, model, metrics
