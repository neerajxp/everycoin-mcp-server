"""
Feature Engineering — computes technical indicators for all historical rows.

Two modes:
  run_feature_engineering()       — incremental: only new rows since last compute
  run_feature_engineering_full()  — full rebuild: recomputes all rows (used after backfill)

Features per row:
  Returns     : 1h, 6h, 24h price returns
  SMA         : 7-period, 24-period simple moving average
  EMA         : 12-period, 26-period exponential moving average
  MACD        : 12-26 EMA diff, 9-period signal, histogram
  RSI         : 14-period relative strength index
  Bollinger   : 20-period bands (upper, middle, lower, width)
  Volatility  : rolling 24-period std of 1h returns
"""

import logging
import sqlite3

import pandas as pd

from mlops import db
from mlops.config import COINS, DB_PATH

log = logging.getLogger("everycoin.mlops.features")


# ── Public entry points ───────────────────────────────────────────────────────

def run_feature_engineering() -> dict[str, int]:
    """Incremental: compute features only for the latest tick (live pipeline use)."""
    log.info("=== Feature engineering (incremental) ===")
    written = 0
    for coin_id in COINS:
        rows = db.price_history(coin_id, limit=200)
        if len(rows) < 2:
            log.warning("  skip %s — not enough data (%d rows)", coin_id, len(rows))
            continue
        df = _build_df(rows)
        row = _row_to_dict(df, coin_id, -1)
        if row:
            db.insert_features(coin_id, row)
            written += 1
            _log_row(row)
    log.info("=== done — %d coins written ===", written)
    return {"coins_processed": written}


def run_feature_engineering_full() -> dict[str, int]:
    """Full rebuild: compute features for every historical row per coin.
    Clears feature_store first, then recomputes from all price_history rows.
    Use this after a backfill."""
    log.info("=== Feature engineering FULL REBUILD ===")
    _clear_feature_store()
    total = 0
    for coin_id in COINS:
        rows = db.price_history(coin_id, limit=10000)
        if len(rows) < 14:
            log.warning("  skip %s — only %d rows", coin_id, len(rows))
            continue
        df    = _build_df(rows)
        count = _insert_all_rows(df, coin_id)
        total += count
        log.info("  ✓ %-15s %d feature rows written", coin_id, count)
    log.info("=== Full rebuild done — %d total rows ===", total)
    return {"total_rows": total}


# ── DataFrame builder ─────────────────────────────────────────────────────────

def _build_df(rows: list[dict]) -> pd.DataFrame:
    """Build chronological price DataFrame with all indicators computed."""
    df = (
        pd.DataFrame(rows)
        .sort_values("fetched_at")
        .reset_index(drop=True)
    )
    df["price_usd"]  = pd.to_numeric(df["price_usd"],  errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["price_usd"]).reset_index(drop=True)

    p = df["price_usd"]

    # Returns
    df["return_1h"]  = p.pct_change(1)
    df["return_6h"]  = p.pct_change(6)
    df["return_24h"] = p.pct_change(24)

    # SMA
    df["sma_7"]  = p.rolling(7).mean()
    df["sma_24"] = p.rolling(24).mean()

    # EMA
    df["ema_12"] = p.ewm(span=12, adjust=False).mean()
    df["ema_26"] = p.ewm(span=26, adjust=False).mean()

    # MACD
    macd_line       = df["ema_12"] - df["ema_26"]
    signal          = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]       = macd_line
    df["macd_signal"] = signal
    df["macd_hist"]  = macd_line - signal

    # RSI (14)
    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20)
    bb_mid        = p.rolling(20).mean()
    bb_std        = p.rolling(20).std()
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # Volatility
    df["volatility_24h"] = df["return_1h"].rolling(24).std()

    return df


# ── Writers ───────────────────────────────────────────────────────────────────

def _insert_all_rows(df: pd.DataFrame, coin_id: str) -> int:
    """Bulk insert all rows from df into feature_store. Returns count inserted."""
    feature_cols = [
        "price_usd", "return_1h", "return_6h", "return_24h",
        "sma_7", "sma_24", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "volatility_24h", "market_cap",
    ]
    # Only keep rows with at least RSI available (needs 14 periods)
    valid = df.dropna(subset=["rsi_14"])
    count = 0
    conn = sqlite3.connect(DB_PATH)
    try:
        for _, row in valid.iterrows():
            features = {col: _r(row.get(col)) for col in feature_cols}
            features["price_usd"]  = _r(row["price_usd"])
            features["market_cap"] = _r(row.get("market_cap"))
            conn.execute("""
                INSERT INTO feature_store (
                    computed_at, coin_id, price_usd,
                    return_1h, return_6h, return_24h,
                    sma_7, sma_24, ema_12, ema_26,
                    macd, macd_signal, macd_hist,
                    rsi_14, bb_upper, bb_middle, bb_lower, bb_width,
                    volatility_24h, market_cap
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row["fetched_at"], coin_id, features["price_usd"],
                features["return_1h"], features["return_6h"], features["return_24h"],
                features["sma_7"], features["sma_24"],
                features["ema_12"], features["ema_26"],
                features["macd"], features["macd_signal"], features["macd_hist"],
                features["rsi_14"],
                features["bb_upper"], features["bb_middle"], features["bb_lower"], features["bb_width"],
                features["volatility_24h"], features["market_cap"],
            ))
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


def _clear_feature_store() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM feature_store")
    conn.commit()
    conn.close()
    log.info("  feature_store cleared")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_dict(df: pd.DataFrame, coin_id: str, idx: int) -> dict | None:
    if df.empty:
        return None
    row = df.iloc[idx]
    cols = [
        "price_usd", "return_1h", "return_6h", "return_24h",
        "sma_7", "sma_24", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist", "rsi_14",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "volatility_24h", "market_cap",
    ]
    return {col: _r(row.get(col)) for col in cols}


def _log_row(row: dict) -> None:
    log.info(
        "  ✓ price=$%.2f rsi=%.1f macd=%.4f return_1h=%.3f%%",
        row.get("price_usd") or 0,
        row.get("rsi_14") or 0,
        row.get("macd") or 0,
        (row.get("return_1h") or 0) * 100,
    )


def _r(val) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        return round(float(val), 6)
    except Exception:
        return None
