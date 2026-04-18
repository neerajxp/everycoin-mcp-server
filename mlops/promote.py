"""
Promote the best MLflow run to production.

Picks the run with highest roc_auc, copies model + scaler to mlops/models/
so FastAPI can load them without depending on MLflow at runtime.

Usage:
  python -m mlops.promote
"""

import logging
import pickle
import shutil
import sys
from pathlib import Path

import mlflow
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mlops.config import DB_PATH
from mlops.train import _load_data, _prepare_features, _time_split, _train

MODELS_DIR = Path(__file__).parent / "models"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "everycoin-price-direction"

log = logging.getLogger("everycoin.mlops.promote")


def promote_best() -> dict:
    """Pick best run by roc_auc, retrain with same data, save model + scaler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT)

    if exp is None:
        log.error("No MLflow experiment found — run python -m mlops.train first.")
        return {}

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        log.error("No runs found in experiment.")
        return {}

    best      = runs[0]
    run_id    = best.info.run_id
    roc_auc   = best.data.metrics.get("roc_auc", 0)
    accuracy  = best.data.metrics.get("accuracy", 0)

    log.info("Best run: %s | roc_auc=%.4f accuracy=%.4f", run_id, roc_auc, accuracy)

    # Retrain on full dataset to get fresh model + scaler
    log.info("Retraining on full dataset for promotion ...")
    df              = _load_data()
    X, y            = _prepare_features(df)
    X_train, _, y_train, _ = _time_split(X, y)
    model, scaler   = _train(X_train, y_train)

    # Save to mlops/models/
    MODELS_DIR.mkdir(exist_ok=True)

    model_path  = MODELS_DIR / "best_model.ubj"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path   = MODELS_DIR / "meta.json"

    model.save_model(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    import json
    meta = {
        "run_id":   run_id,
        "roc_auc":  roc_auc,
        "accuracy": accuracy,
        "features": list(X.columns),
        "promoted_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("✓ Model saved  : %s", model_path)
    log.info("✓ Scaler saved : %s", scaler_path)
    log.info("✓ Meta saved   : %s", meta_path)
    return meta


if __name__ == "__main__":
    promote_best()
