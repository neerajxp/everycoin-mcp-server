"""
Backfill historical price data from CoinGecko market_chart API.

Fetches up to 90 days of hourly price + market cap data per coin
and inserts into price_history, skipping rows already present.

Usage:
    python -m mlops.backfill            # 90 days (default)
    python -m mlops.backfill --days 30  # 30 days
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from mlops import db
from mlops.config import COINS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("everycoin.mlops.backfill")


async def backfill_coin(client: httpx.AsyncClient, coin_id: str, days: int) -> int:
    """Fetch historical data for one coin. Returns number of rows inserted."""
    try:
        res = await client.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": str(days)},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()

        prices      = data.get("prices", [])
        market_caps = {row[0]: row[1] for row in data.get("market_caps", [])}

        inserted = 0
        for ts_ms, price_usd in prices:
            fetched_at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")
            market_cap = market_caps.get(ts_ms)
            db.insert_price_at(
                fetched_at=fetched_at,
                coin_id=coin_id,
                price_usd=price_usd,
                change_24h=None,   # not available in market_chart
                market_cap=market_cap,
            )
            inserted += 1

        log.info("  ✓ %-15s %d rows inserted (%d days)", coin_id, inserted, days)
        return inserted

    except Exception as e:
        log.error("  ✗ %s failed: %s", coin_id, e)
        return 0


async def run_backfill(days: int) -> None:
    log.info("=== Backfill started — %d days of history ===", days)
    db.init_db()
    total = 0

    async with httpx.AsyncClient() as client:
        for coin_id in COINS:
            n = await backfill_coin(client, coin_id, days)
            total += n
            # CoinGecko free tier: ~10 req/min — small delay between coins
            time.sleep(1.5)

    log.info("=== Backfill done — %d total rows inserted ===", total)
    counts = db.row_counts()
    log.info("DB row counts: %s", counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical price data")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch (max 90 free tier)")
    args = parser.parse_args()
    asyncio.run(run_backfill(args.days))


if __name__ == "__main__":
    main()
