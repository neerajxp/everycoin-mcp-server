"""
SQLite database setup and helpers for the MLOps pipeline.

Tables:
  price_history  — hourly price snapshots per coin
  gas_history    — hourly Ethereum gas snapshots
  defi_history   — hourly DeFi protocol TVL snapshots
  feature_store  — engineered features per coin per timestamp
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from mlops.config import DB_PATH

log = logging.getLogger("everycoin.mlops.db")


# ── Setup ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at    TEXT    NOT NULL,
                coin_id       TEXT    NOT NULL,
                price_usd     REAL,
                change_24h    REAL,
                market_cap    REAL
            );

            CREATE INDEX IF NOT EXISTS idx_price_coin
                ON price_history (coin_id, fetched_at);

            CREATE TABLE IF NOT EXISTS gas_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at    TEXT    NOT NULL,
                slow_gwei     REAL,
                standard_gwei REAL,
                fast_gwei     REAL
            );

            CREATE TABLE IF NOT EXISTS defi_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at    TEXT    NOT NULL,
                protocol      TEXT    NOT NULL,
                tvl_usd       REAL,
                category      TEXT,
                chains        TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_defi_protocol
                ON defi_history (protocol, fetched_at);

            CREATE TABLE IF NOT EXISTS feature_store (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at     TEXT    NOT NULL,
                coin_id         TEXT    NOT NULL,
                price_usd       REAL,
                return_1h       REAL,
                return_6h       REAL,
                return_24h      REAL,
                sma_7           REAL,
                sma_24          REAL,
                ema_12          REAL,
                ema_26          REAL,
                macd            REAL,
                macd_signal     REAL,
                macd_hist       REAL,
                rsi_14          REAL,
                bb_upper        REAL,
                bb_middle       REAL,
                bb_lower        REAL,
                bb_width        REAL,
                volatility_24h  REAL,
                market_cap      REAL
            );

            CREATE INDEX IF NOT EXISTS idx_feature_coin
                ON feature_store (coin_id, computed_at);
        """)
    log.info("DB ready: %s", DB_PATH)


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Writers ───────────────────────────────────────────────────────────────────

def insert_price(coin_id: str, price_usd: float | None, change_24h: float | None, market_cap: float | None) -> None:
    insert_price_at(_now(), coin_id, price_usd, change_24h, market_cap)


def insert_price_at(fetched_at: str, coin_id: str, price_usd: float | None, change_24h: float | None, market_cap: float | None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO price_history (fetched_at, coin_id, price_usd, change_24h, market_cap) VALUES (?,?,?,?,?)",
            (fetched_at, coin_id, price_usd, change_24h, market_cap),
        )
    log.debug("  ✓ price saved: %s $%.2f @ %s", coin_id, price_usd or 0, fetched_at)


def insert_gas(slow: float | None, standard: float | None, fast: float | None) -> None:
    ts = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO gas_history (fetched_at, slow_gwei, standard_gwei, fast_gwei) VALUES (?,?,?,?)",
            (ts, slow, standard, fast),
        )
    log.debug("  ✓ gas saved: slow=%s std=%s fast=%s", slow, standard, fast)


def insert_defi(protocol: str, tvl_usd: float | None, category: str | None, chains: list[str]) -> None:
    ts = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO defi_history (fetched_at, protocol, tvl_usd, category, chains) VALUES (?,?,?,?,?)",
            (ts, protocol, tvl_usd, category, ",".join(chains or [])),
        )
    log.debug("  ✓ defi saved: %s tvl=$%.0f", protocol, tvl_usd or 0)


# ── Readers ───────────────────────────────────────────────────────────────────

def latest_prices(limit: int = 100) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM price_history ORDER BY fetched_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def price_history(coin_id: str, limit: int = 168) -> list[dict]:
    """Return up to `limit` rows for one coin (default = last 7 days hourly)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM price_history WHERE coin_id=? ORDER BY fetched_at DESC LIMIT ?",
            (coin_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_features(coin_id: str, features: dict) -> None:
    ts = _now()
    with _conn() as conn:
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
            ts, coin_id, features.get("price_usd"),
            features.get("return_1h"), features.get("return_6h"), features.get("return_24h"),
            features.get("sma_7"), features.get("sma_24"),
            features.get("ema_12"), features.get("ema_26"),
            features.get("macd"), features.get("macd_signal"), features.get("macd_hist"),
            features.get("rsi_14"),
            features.get("bb_upper"), features.get("bb_middle"), features.get("bb_lower"), features.get("bb_width"),
            features.get("volatility_24h"), features.get("market_cap"),
        ))


def latest_features(coin_id: str | None = None) -> list[dict]:
    """Return the most recent feature row per coin (or for one specific coin)."""
    with _conn() as conn:
        if coin_id:
            rows = conn.execute(
                "SELECT * FROM feature_store WHERE coin_id=? ORDER BY computed_at DESC LIMIT 1",
                (coin_id,),
            ).fetchall()
        else:
            rows = conn.execute("""
                SELECT f.* FROM feature_store f
                INNER JOIN (
                    SELECT coin_id, MAX(computed_at) AS latest
                    FROM feature_store GROUP BY coin_id
                ) m ON f.coin_id = m.coin_id AND f.computed_at = m.latest
                ORDER BY f.coin_id
            """).fetchall()
    return [dict(r) for r in rows]


def row_counts() -> dict[str, int]:
    with _conn() as conn:
        return {
            "price_history": conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0],
            "gas_history":   conn.execute("SELECT COUNT(*) FROM gas_history").fetchone()[0],
            "defi_history":  conn.execute("SELECT COUNT(*) FROM defi_history").fetchone()[0],
            "feature_store": conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
