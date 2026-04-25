"""
MySQL database setup and helpers for the MLOps pipeline.
Replaces the SQLite implementation — public API is identical so no callers change.

Connection is configured via environment variables (set in Railway dashboard):
    MYSQL_HOST      — e.g. srv1234.hstgr.io
    MYSQL_PORT      — default 3306
    MYSQL_USER      — your Hostinger MySQL username
    MYSQL_PASSWORD  — your Hostinger MySQL password
    MYSQL_DATABASE  — database name (create it once in Hostinger hPanel)
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import pymysql
import pymysql.cursors

log = logging.getLogger("everycoin.mlops.db")

# ── Connection config from env ────────────────────────────────────────────────

def _cfg() -> dict:
    return {
        "host":     os.environ["MYSQL_HOST"],
        "port":     int(os.getenv("MYSQL_PORT", "3306")),
        "user":     os.environ["MYSQL_USER"],
        "password": os.environ["MYSQL_PASSWORD"],
        "database": os.environ["MYSQL_DATABASE"],
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": False,
        "charset": "utf8mb4",
    }


@contextmanager
def _conn() -> Generator[pymysql.connections.Connection, None, None]:
    conn = pymysql.connect(**_cfg())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Setup ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    fetched_at  VARCHAR(32)  NOT NULL,
                    coin_id     VARCHAR(64)  NOT NULL,
                    price_usd   DOUBLE,
                    change_24h  DOUBLE,
                    market_cap  DOUBLE,
                    INDEX idx_price_coin (coin_id, fetched_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gas_history (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    fetched_at    VARCHAR(32) NOT NULL,
                    slow_gwei     DOUBLE,
                    standard_gwei DOUBLE,
                    fast_gwei     DOUBLE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS defi_history (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    fetched_at  VARCHAR(32)  NOT NULL,
                    protocol    VARCHAR(64)  NOT NULL,
                    tvl_usd     DOUBLE,
                    category    VARCHAR(64),
                    chains      VARCHAR(255),
                    INDEX idx_defi_protocol (protocol, fetched_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feature_store (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    computed_at     VARCHAR(32)  NOT NULL,
                    coin_id         VARCHAR(64)  NOT NULL,
                    price_usd       DOUBLE,
                    return_1h       DOUBLE,
                    return_6h       DOUBLE,
                    return_24h      DOUBLE,
                    sma_7           DOUBLE,
                    sma_24          DOUBLE,
                    ema_12          DOUBLE,
                    ema_26          DOUBLE,
                    macd            DOUBLE,
                    macd_signal     DOUBLE,
                    macd_hist       DOUBLE,
                    rsi_14          DOUBLE,
                    bb_upper        DOUBLE,
                    bb_middle       DOUBLE,
                    bb_lower        DOUBLE,
                    bb_width        DOUBLE,
                    volatility_24h  DOUBLE,
                    market_cap      DOUBLE,
                    INDEX idx_feature_coin (coin_id, computed_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id                  INT AUTO_INCREMENT PRIMARY KEY,
                    predicted_at        VARCHAR(32)  NOT NULL,
                    coin_id             VARCHAR(64)  NOT NULL,
                    current_price       DOUBLE       NOT NULL,
                    predicted_price     DOUBLE       NOT NULL,
                    predicted_move_pct  DOUBLE       NOT NULL,
                    predicted_score     INT          NOT NULL,
                    confidence          DOUBLE       NOT NULL,
                    actual_price        DOUBLE,
                    outcome             VARCHAR(32),
                    INDEX idx_pred_coin (coin_id, predicted_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
    log.info("MySQL DB ready: %s@%s/%s", os.getenv("MYSQL_USER"), os.getenv("MYSQL_HOST"), os.getenv("MYSQL_DATABASE"))


# ── Writers ───────────────────────────────────────────────────────────────────

def insert_price(coin_id: str, price_usd: float | None, change_24h: float | None, market_cap: float | None) -> None:
    insert_price_at(_now(), coin_id, price_usd, change_24h, market_cap)


def insert_price_at(fetched_at: str, coin_id: str, price_usd: float | None, change_24h: float | None, market_cap: float | None) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO price_history (fetched_at, coin_id, price_usd, change_24h, market_cap) VALUES (%s,%s,%s,%s,%s)",
                (fetched_at, coin_id, price_usd, change_24h, market_cap),
            )
    log.debug("  ✓ price saved: %s $%.2f @ %s", coin_id, price_usd or 0, fetched_at)


def insert_gas(slow: float | None, standard: float | None, fast: float | None) -> None:
    ts = _now()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO gas_history (fetched_at, slow_gwei, standard_gwei, fast_gwei) VALUES (%s,%s,%s,%s)",
                (ts, slow, standard, fast),
            )
    log.debug("  ✓ gas saved: slow=%s std=%s fast=%s", slow, standard, fast)


def insert_defi(protocol: str, tvl_usd: float | None, category: str | None, chains: list[str]) -> None:
    ts = _now()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO defi_history (fetched_at, protocol, tvl_usd, category, chains) VALUES (%s,%s,%s,%s,%s)",
                (ts, protocol, tvl_usd, category, ",".join(chains or [])),
            )
    log.debug("  ✓ defi saved: %s tvl=$%.0f", protocol, tvl_usd or 0)


def insert_features_at(computed_at: str, coin_id: str, features: dict) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_store (
                    computed_at, coin_id, price_usd,
                    return_1h, return_6h, return_24h,
                    sma_7, sma_24, ema_12, ema_26,
                    macd, macd_signal, macd_hist,
                    rsi_14, bb_upper, bb_middle, bb_lower, bb_width,
                    volatility_24h, market_cap
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                computed_at, coin_id, features.get("price_usd"),
                features.get("return_1h"), features.get("return_6h"), features.get("return_24h"),
                features.get("sma_7"), features.get("sma_24"),
                features.get("ema_12"), features.get("ema_26"),
                features.get("macd"), features.get("macd_signal"), features.get("macd_hist"),
                features.get("rsi_14"),
                features.get("bb_upper"), features.get("bb_middle"), features.get("bb_lower"), features.get("bb_width"),
                features.get("volatility_24h"), features.get("market_cap"),
            ))


def clear_feature_store() -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM feature_store")
    log.info("feature_store cleared")


def insert_features(coin_id: str, features: dict) -> None:
    ts = _now()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_store (
                    computed_at, coin_id, price_usd,
                    return_1h, return_6h, return_24h,
                    sma_7, sma_24, ema_12, ema_26,
                    macd, macd_signal, macd_hist,
                    rsi_14, bb_upper, bb_middle, bb_lower, bb_width,
                    volatility_24h, market_cap
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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


def insert_price_prediction(coin_id: str, current_price: float, predicted_price: float,
                             predicted_move_pct: float, predicted_score: int, confidence: float) -> int:
    ts = _now()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO price_predictions
                    (predicted_at, coin_id, current_price, predicted_price,
                     predicted_move_pct, predicted_score, confidence)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (ts, coin_id, current_price, predicted_price, predicted_move_pct, predicted_score, confidence))
            row_id = cur.lastrowid
    log.info("prediction saved: %s target=$%.2f move=%.2f%% id=%s", coin_id, predicted_price, predicted_move_pct, row_id)
    return row_id


def update_prediction_outcome(prediction_id: int, actual_price: float) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT predicted_price FROM price_predictions WHERE id=%s", (prediction_id,)
            )
            row = cur.fetchone()
            if not row:
                return
            predicted = row["predicted_price"]
            diff_pct = abs(actual_price - predicted) / predicted * 100
            outcome = "Hit" if diff_pct <= 3.0 else f"Miss {diff_pct:.1f}%"
            cur.execute(
                "UPDATE price_predictions SET actual_price=%s, outcome=%s WHERE id=%s",
                (actual_price, outcome, prediction_id),
            )
    log.info("outcome updated id=%d actual=$%.2f outcome=%s", prediction_id, actual_price, outcome)


# ── Readers ───────────────────────────────────────────────────────────────────

def latest_prices(limit: int = 100) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM price_history ORDER BY fetched_at DESC LIMIT %s", (limit,))
            return cur.fetchall()


def price_history(coin_id: str, limit: int = 168) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM price_history WHERE coin_id=%s ORDER BY fetched_at DESC LIMIT %s",
                (coin_id, limit),
            )
            return cur.fetchall()


def latest_features(coin_id: str | None = None) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            if coin_id:
                cur.execute(
                    "SELECT * FROM feature_store WHERE coin_id=%s ORDER BY computed_at DESC LIMIT 1",
                    (coin_id,),
                )
                return cur.fetchall()
            else:
                cur.execute("""
                    SELECT f.* FROM feature_store f
                    INNER JOIN (
                        SELECT coin_id, MAX(computed_at) AS latest
                        FROM feature_store GROUP BY coin_id
                    ) m ON f.coin_id = m.coin_id AND f.computed_at = m.latest
                    ORDER BY f.coin_id
                """)
                return cur.fetchall()


def latest_prediction(coin_id: str) -> dict | None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM price_predictions WHERE coin_id=%s ORDER BY predicted_at DESC LIMIT 1",
                (coin_id,),
            )
            return cur.fetchone()


def prediction_history(coin_id: str, limit: int = 30) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM price_predictions WHERE coin_id=%s ORDER BY predicted_at DESC LIMIT %s",
                (coin_id, limit),
            )
            return cur.fetchall()


def price_history_range(coin_id: str, start: str, end: str) -> list[dict]:
    """Return hourly price rows between two ISO timestamps (inclusive)."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fetched_at, price_usd FROM price_history "
                "WHERE coin_id=%s AND fetched_at >= %s AND fetched_at <= %s "
                "ORDER BY fetched_at ASC",
                (coin_id, start, end),
            )
            return cur.fetchall()


def available_forecast_dates(coin_id: str, limit: int = 7) -> list[str]:
    """Return distinct YYYY-MM-DD strings for days that have predictions, newest first."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT LEFT(predicted_at, 10) AS date FROM price_predictions "
                "WHERE coin_id=%s ORDER BY date DESC LIMIT %s",
                (coin_id, limit),
            )
            return [row["date"] for row in cur.fetchall()]


def prediction_for_date(coin_id: str, date_str: str) -> dict | None:
    """Return the latest prediction for a given YYYY-MM-DD date."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM price_predictions WHERE coin_id=%s AND LEFT(predicted_at,10)=%s "
                "ORDER BY predicted_at DESC LIMIT 1",
                (coin_id, date_str),
            )
            return cur.fetchone()


def row_counts() -> dict[str, int]:
    with _conn() as conn:
        with conn.cursor() as cur:
            result = {}
            for table in ("price_history", "gas_history", "defi_history", "feature_store", "price_predictions"):
                cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                result[table] = cur.fetchone()["n"]
            return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
