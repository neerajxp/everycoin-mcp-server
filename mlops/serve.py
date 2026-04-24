"""
Model serving — loads best_model.ubj and scaler.pkl,
reads latest features from feature_store, returns AI score + direction.

AI Score (0-100):
  Derived from model's UP probability × 100
  75-100 → BUY  |  45-74 → HOLD  |  0-44 → SELL
"""

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from xgboost import XGBClassifier

from mlops.config import COINS
from mlops import db

log = logging.getLogger("everycoin.mlops.serve")

MODELS_DIR  = Path(__file__).parent / "models"
MODEL_PATH  = MODELS_DIR / "best_model.ubj"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
META_PATH   = MODELS_DIR / "meta.json"

FEATURE_COLS = [
    "return_1h", "return_6h", "return_24h",
    "sma_7", "sma_24", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "volatility_24h",
]


# ── Singleton loader ──────────────────────────────────────────────────────────

_model  = None
_scaler = None
_meta   = {}


def _load_model():
    global _model, _scaler, _meta
    if _model is not None:
        return

    if not MODEL_PATH.exists():
        hf_token   = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID")
        if hf_token and hf_repo_id:
            log.info("Model not found locally — downloading from HF: %s", hf_repo_id)
            MODELS_DIR.mkdir(exist_ok=True)
            for filename in ("best_model.ubj", "scaler.pkl", "meta.json"):
                hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=filename,
                    local_dir=str(MODELS_DIR),
                    token=hf_token,
                )
                log.info("  ✓ downloaded %s", filename)
        else:
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH} and HF_TOKEN/HF_REPO_ID not set. "
                "Run: python -m mlops.promote"
            )

    _model = XGBClassifier()
    _model.load_model(str(MODEL_PATH))

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    if META_PATH.exists():
        with open(META_PATH) as f:
            _meta = json.load(f)

    log.info("Model loaded — roc_auc=%.4f promoted_at=%s",
             _meta.get("roc_auc", 0), _meta.get("promoted_at", "unknown"))


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(coin_id: str) -> dict:
    """
    Return AI score, direction, and confidence for a coin.

    Returns:
      {
        "coin_id":    "bitcoin",
        "ai_score":   72,
        "direction":  "HOLD",
        "confidence": 0.72,
        "signal":     { rsi_14, macd, return_1h, ... }
        "error":      None | str
      }
    """
    _load_model()

    # Get latest features for this coin
    rows = db.latest_features(coin_id)
    if not rows:
        return _error(coin_id, "No features found — run scheduler first")

    row = rows[0]

    # Build feature vector — fill missing with 0
    feature_values = [row.get(col) or 0.0 for col in FEATURE_COLS]

    # Check if we have meaningful data
    if all(v == 0.0 for v in feature_values):
        return _error(coin_id, "All features are zero — insufficient data")

    X = pd.DataFrame([feature_values], columns=FEATURE_COLS)
    X_scaled = _scaler.transform(X)

    prob_up   = float(_model.predict_proba(X_scaled)[0][1])
    ai_score  = round(prob_up * 100)
    direction = _score_to_direction(ai_score)

    return {
        "coin_id":    coin_id,
        "ai_score":   ai_score,
        "direction":  direction,
        "confidence": round(prob_up, 4),
        "signal": {
            "rsi_14":     round(row.get("rsi_14") or 0, 2),
            "macd":       round(row.get("macd") or 0, 4),
            "return_1h":  round((row.get("return_1h") or 0) * 100, 3),
            "return_24h": round((row.get("return_24h") or 0) * 100, 3),
            "bb_width":   round(row.get("bb_width") or 0, 4),
            "volatility": round(row.get("volatility_24h") or 0, 6),
        },
        "model_roc_auc": _meta.get("roc_auc"),
        "error": None,
    }


def predict_all() -> list[dict]:
    """Return predictions for all tracked coins."""
    return [predict(coin_id) for coin_id in COINS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_to_direction(score: int) -> str:
    if score >= 75:
        return "BUY"
    if score >= 45:
        return "HOLD"
    return "SELL"


def _error(coin_id: str, msg: str) -> dict:
    log.warning("predict(%s): %s", coin_id, msg)
    return {
        "coin_id":    coin_id,
        "ai_score":   50,
        "direction":  "HOLD",
        "confidence": 0.5,
        "signal":     {},
        "model_roc_auc": None,
        "error": msg,
    }
