"""
Promote trained model to production.

Trains on full dataset, saves model + scaler locally, pushes to Hugging Face.
Generates a README.md (Model Card) with metrics visible on HF.

Usage:
  python -m mlops.promote
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi

from mlops.train import _load_data, _prepare_features, _time_split, _train, _evaluate, FEATURE_COLS

MODELS_DIR = Path(__file__).parent / "models"

log = logging.getLogger("everycoin.mlops.promote")


def _generate_readme(meta: dict) -> str:
    promoted_at = meta.get("promoted_at", "unknown")
    roc_auc     = meta.get("roc_auc", 0)
    accuracy    = meta.get("accuracy", 0)
    f1          = meta.get("f1", 0)
    precision   = meta.get("precision", 0)
    recall      = meta.get("recall", 0)
    features    = meta.get("features", FEATURE_COLS)

    return f"""---
license: mit
tags:
  - xgboost
  - finance
  - cryptocurrency
  - price-prediction
metrics:
  - roc_auc
  - accuracy
---

# EveryCoin Price Direction Model

XGBoost binary classifier predicting whether a cryptocurrency price will go **UP** in the next hour.

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Task | Binary classification (UP / DOWN) |
| Coins trained on | BTC, ETH, SOL, BNB, ADA, AVAX, LINK, UNI |
| Training data | 90 days of hourly price history |
| Promoted at | {promoted_at} |

## Evaluation Metrics

| Metric | Score |
|---|---|
| ROC-AUC | {roc_auc:.4f} |
| Accuracy | {accuracy:.4f} |
| F1 Score | {f1:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |

## Features ({len(features)})

{", ".join(f"`{f}`" for f in features)}

## Files

| File | Description |
|---|---|
| `best_model.ubj` | XGBoost model binary |
| `scaler.pkl` | StandardScaler fitted on training data |
| `meta.json` | Metrics and metadata |
"""


def promote_best() -> dict:
    """Train on full dataset, save model + scaler, push to Hugging Face."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    log.info("Loading data ...")
    df = _load_data()
    if df.empty:
        log.error("No data — run backfill and feature engineering first.")
        return {}

    X, y = _prepare_features(df)
    X_train, X_test, y_train, y_test = _time_split(X, y)

    log.info("Training on %d rows ...", len(X_train))
    model, scaler = _train(X_train, y_train)
    metrics       = _evaluate(model, scaler, X_test, y_test)

    log.info("roc_auc=%.4f accuracy=%.4f f1=%.4f", metrics["roc_auc"], metrics["accuracy"], metrics["f1"])

    # Save locally
    MODELS_DIR.mkdir(exist_ok=True)
    model_path  = MODELS_DIR / "best_model.ubj"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path   = MODELS_DIR / "meta.json"

    model.save_model(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "roc_auc":      metrics["roc_auc"],
        "accuracy":     metrics["accuracy"],
        "f1":           metrics["f1"],
        "precision":    metrics["precision"],
        "recall":       metrics["recall"],
        "features":     list(X.columns),
        "promoted_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("✓ Model saved  : %s", model_path)
    log.info("✓ Scaler saved : %s", scaler_path)
    log.info("✓ Meta saved   : %s", meta_path)

    # Push to Hugging Face
    hf_token   = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")
    if hf_token and hf_repo_id:
        log.info("Pushing to Hugging Face: %s ...", hf_repo_id)
        api = HfApi()
        readme = _generate_readme(meta)
        readme_path = MODELS_DIR / "README.md"
        readme_path.write_text(readme)

        for filename, path in [
            ("best_model.ubj", model_path),
            ("scaler.pkl",     scaler_path),
            ("meta.json",      meta_path),
            ("README.md",      readme_path),
        ]:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=hf_repo_id,
                token=hf_token,
                repo_type="model",
            )
            log.info("  ✓ uploaded %s", filename)
        log.info("✓ Model live at: https://huggingface.co/%s", hf_repo_id)
    else:
        log.warning("HF_TOKEN or HF_REPO_ID not set — skipping Hugging Face upload")

    return meta


if __name__ == "__main__":
    promote_best()
