"""
SQLite database setup and helpers for the MLOps pipeline.

Tables:
  price_history  — hourly price snapshots per coin
  gas_history    — hourly Ethereum gas snapshots
  defi_history   — hourly DeFi protocol TVL snapshots
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
    ts = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO price_history (fetched_at, coin_id, price_usd, change_24h, market_cap) VALUES (?,?,?,?,?)",
            (ts, coin_id, price_usd, change_24h, market_cap),
        )
    log.debug("  ✓ price saved: %s $%.2f", coin_id, price_usd or 0)


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


def row_counts() -> dict[str, int]:
    with _conn() as conn:
        return {
            "price_history": conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0],
            "gas_history":   conn.execute("SELECT COUNT(*) FROM gas_history").fetchone()[0],
            "defi_history":  conn.execute("SELECT COUNT(*) FROM defi_history").fetchone()[0],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
