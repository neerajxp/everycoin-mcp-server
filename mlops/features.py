"""
Week 2 — Feature Engineering.

Reads price_history from SQLite, computes technical indicators using pandas,
and writes the result to feature_store.

Features computed per coin:
  Returns     : 1h, 6h, 24h price returns
  SMA         : 7-period, 24-period simple moving average
  EMA         : 12-period, 26-period exponential moving average
  MACD        : 12-26 EMA diff, 9-period signal, histogram
  RSI         : 14-period relative strength index
  Bollinger   : 20-period bands (upper, middle, lower, width)
  Volatility  : rolling 24-period std of 1h returns
"""

import logging
import pandas as pd

from mlops import db
from mlops.config import COINS

log = logging.getLogger("everycoin.mlops.features")


# ── Main entry point ─────────────────────────────────────────────────────────

def run_feature_engineering() -> dict[str, int]:
    """Compute and store features for all coins. Returns count of rows written."""
    log.info("=== Feature engineering started ===")
    written = 0
    for coin_id in COINS:
        rows = db.price_history(coin_id, limit=200)
        if len(rows) < 2:
            log.warning("  skip %s — not enough data (%d rows)", coin_id, len(rows))
            continue

        features = _compute_features(coin_id, rows)
        if features:
            db.insert_features(coin_id, features)
            written += 1
            log.info(
                "  ✓ %s | price=$%.2f rsi=%.1f macd=%.4f return_1h=%.2f%%",
                coin_id,
                features.get("price_usd") or 0,
                features.get("rsi_14") or 0,
                features.get("macd") or 0,
                (features.get("return_1h") or 0) * 100,
            )

    log.info("=== Feature engineering done — %d coins processed ===", written)
    return {"coins_processed": written}


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_features(coin_id: str, rows: list[dict]) -> dict | None:
    # rows come back DESC from db — reverse to get chronological order
    df = pd.DataFrame(rows).sort_values("fetched_at").reset_index(drop=True)
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["price_usd"])

    if len(df) < 2:
        return None

    price = df["price_usd"]
    n = len(price)

    features: dict = {
        "price_usd":  _last(price),
        "market_cap": _last(pd.to_numeric(df["market_cap"], errors="coerce")),
    }

    # ── Returns ───────────────────────────────────────────────────────────────
    features["return_1h"]  = _pct_change(price, 1)
    features["return_6h"]  = _pct_change(price, 6)
    features["return_24h"] = _pct_change(price, 24)

    # ── SMA ───────────────────────────────────────────────────────────────────
    features["sma_7"]  = _sma(price, min(7, n))
    features["sma_24"] = _sma(price, min(24, n))

    # ── EMA ───────────────────────────────────────────────────────────────────
    features["ema_12"] = _ema(price, min(12, n))
    features["ema_26"] = _ema(price, min(26, n))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = price.ewm(span=min(12, n), adjust=False).mean()
    ema26 = price.ewm(span=min(26, n), adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=min(9, n), adjust=False).mean()
    features["macd"]        = _last(macd_line)
    features["macd_signal"] = _last(signal)
    features["macd_hist"]   = _r(_last(macd_line) - _last(signal))

    # ── RSI ───────────────────────────────────────────────────────────────────
    features["rsi_14"] = _rsi(price, min(14, n))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    window = min(20, n)
    bb_mid   = price.rolling(window).mean()
    bb_std   = price.rolling(window).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    mid_val  = _last(bb_mid)
    up_val   = _last(bb_upper)
    lo_val   = _last(bb_lower)
    features["bb_upper"]  = up_val
    features["bb_middle"] = mid_val
    features["bb_lower"]  = lo_val
    features["bb_width"]  = _r((up_val - lo_val) / mid_val) if mid_val else None

    # ── Volatility ────────────────────────────────────────────────────────────
    returns = price.pct_change()
    features["volatility_24h"] = _r(returns.rolling(min(24, n)).std().iloc[-1])

    return features


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _sma(price: pd.Series, window: int) -> float | None:
    val = price.rolling(window).mean().iloc[-1]
    return _r(val)


def _ema(price: pd.Series, span: int) -> float | None:
    val = price.ewm(span=span, adjust=False).mean().iloc[-1]
    return _r(val)


def _rsi(price: pd.Series, window: int) -> float | None:
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return _r(val)


def _pct_change(price: pd.Series, periods: int) -> float | None:
    if len(price) <= periods:
        return None
    prev = price.iloc[-(periods + 1)]
    curr = price.iloc[-1]
    if prev == 0:
        return None
    return _r((curr - prev) / prev)


def _last(series: pd.Series) -> float | None:
    val = series.iloc[-1]
    return _r(val)


def _r(val: float | None) -> float | None:
    """Round to 6 decimal places, return None for NaN/inf."""
    try:
        if val is None or pd.isna(val) or not pd.api.types.is_float(val):
            return None
        return round(float(val), 6)
    except Exception:
        return None
