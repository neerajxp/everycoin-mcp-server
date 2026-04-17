"""
Week 3 — Model Training with XGBoost + MLflow.

Task: Binary classification — will price go UP in the next hour?
  Label 1 = next hour price > current price
  Label 0 = next hour price <= current price

Pipeline:
  1. Load feature_store from SQLite
  2. Create target label (next-hour return > 0)
  3. Train/test split (80/20, time-ordered — no data leakage)
  4. Train XGBoost classifier
  5. Evaluate: accuracy, F1, precision, recall, ROC-AUC
  6. Log everything to MLflow (params, metrics, model artifact)

Usage:
  python -m mlops.train                    # train on all coins combined
  python -m mlops.train --coin bitcoin     # train on one coin only
  python -m mlops.train --no-mlflow        # skip MLflow (quick test)
"""

import argparse
import logging
import sys

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mlops.config import DB_PATH, COINS

log = logging.getLogger("everycoin.mlops.train")

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    "return_1h", "return_6h", "return_24h",
    "sma_7", "sma_24", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "volatility_24h",
]

MLFLOW_EXPERIMENT = "everycoin-price-direction"


# ── Main ──────────────────────────────────────────────────────────────────────

def train(coin_id: str | None = None, use_mlflow: bool = True) -> dict:
    """Callable entry point — used by scheduler for weekly retrain."""
    df = _load_data(coin_id=coin_id)
    if df.empty:
        log.error("No data loaded — run backfill and feature engineering first.")
        return {}

    log.info("Loaded %d rows | coins: %s", len(df), sorted(df["coin_id"].unique()))

    X, y        = _prepare_features(df)
    X_train, X_test, y_train, y_test = _time_split(X, y)

    log.info("Train: %d rows | Test: %d rows | Label balance: %.1f%% UP",
             len(X_train), len(X_test), y_train.mean() * 100)

    model, scaler = _train(X_train, y_train)
    metrics       = _evaluate(model, scaler, X_test, y_test)

    _print_metrics(metrics)
    _print_feature_importance(model, X.columns.tolist())

    if use_mlflow:
        _log_to_mlflow(model, scaler, metrics, coin_id)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost price direction classifier")
    parser.add_argument("--coin",      type=str, default=None, help="Train on one coin only (e.g. bitcoin)")
    parser.add_argument("--no-mlflow", action="store_true",    help="Skip MLflow logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    df = _load_data(coin_id=args.coin)
    if df.empty:
        log.error("No data loaded — run backfill and feature engineering first.")
        return

    log.info("Loaded %d rows | coins: %s", len(df), sorted(df["coin_id"].unique()))

    X, y = _prepare_features(df)
    X_train, X_test, y_train, y_test = _time_split(X, y)

    log.info("Train: %d rows | Test: %d rows | Label balance: %.1f%% UP",
             len(X_train), len(X_test), y_train.mean() * 100)

    model, scaler = _train(X_train, y_train)
    metrics       = _evaluate(model, scaler, X_test, y_test)

    _print_metrics(metrics)
    _print_feature_importance(model, X.columns.tolist())

    if not args.no_mlflow:
        _log_to_mlflow(model, scaler, metrics, args.coin)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(coin_id: str | None = None) -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    if coin_id:
        df = pd.read_sql(
            "SELECT * FROM feature_store WHERE coin_id=? ORDER BY computed_at",
            conn, params=(coin_id,)
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM feature_store ORDER BY coin_id, computed_at",
            conn
        )
    conn.close()
    return df


# ── Feature preparation ───────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create label: 1 if next-hour price > current price, 0 otherwise."""
    dfs = []
    for coin_id, group in df.groupby("coin_id"):
        g = group.copy().reset_index(drop=True)
        # shift(-1) gets the NEXT row's price — label for current row
        g["label"] = (g["price_usd"].shift(-1) > g["price_usd"]).astype(int)
        dfs.append(g)

    combined = pd.concat(dfs).reset_index(drop=True)

    # Drop last row per coin (no next price available) and rows missing features
    available = [c for c in FEATURE_COLS if c in combined.columns]
    combined  = combined.dropna(subset=available + ["label"])

    X = combined[available]
    y = combined["label"]
    return X, y


# ── Train / test split ────────────────────────────────────────────────────────

def _time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Time-ordered split — no shuffling to avoid data leakage."""
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ── Training ──────────────────────────────────────────────────────────────────

def _train(X_train: pd.DataFrame, y_train: pd.Series):
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train, verbose=False)
    return model, scaler


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, scaler, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    X_scaled  = scaler.transform(X_test)
    y_pred    = model.predict(X_scaled)
    y_prob    = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "report":    classification_report(y_test, y_pred, target_names=["DOWN", "UP"]),
    }


# ── MLflow logging ────────────────────────────────────────────────────────────

def _log_to_mlflow(model, scaler, metrics: dict, coin_id: str | None) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        # Params
        mlflow.log_param("model",      "XGBoostClassifier")
        mlflow.log_param("coin",       coin_id or "all")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth",    model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("features",   len(FEATURE_COLS))

        # Metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("f1",        metrics["f1"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("roc_auc",   metrics["roc_auc"])

        # Model artifact
        mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        log.info("MLflow run logged: %s", run_id)
        log.info("View UI: mlflow ui  (then open http://localhost:5000)")


# ── Pretty printing ───────────────────────────────────────────────────────────

def _print_metrics(metrics: dict) -> None:
    log.info("")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  EVALUATION RESULTS")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Accuracy  : %.4f", metrics["accuracy"])
    log.info("  F1 Score  : %.4f", metrics["f1"])
    log.info("  Precision : %.4f", metrics["precision"])
    log.info("  Recall    : %.4f", metrics["recall"])
    log.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("\n%s", metrics["report"])


def _print_feature_importance(model, feature_names: list[str]) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    log.info("TOP FEATURES:")
    for feat, score in importance.head(8).items():
        bar = "█" * int(score * 200)
        log.info("  %-18s %.4f  %s", feat, score, bar)


if __name__ == "__main__":
    main()
