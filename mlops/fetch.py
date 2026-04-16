"""
Async data fetchers for the MLOps pipeline.
Mirrors the same API calls in mcp_tools.py but writes results to SQLite
instead of returning them to a LangGraph agent.
"""

import asyncio
import logging

import httpx

from mlops.config import COINS, PROTOCOLS, ETHERSCAN_API_KEY
from mlops import db

log = logging.getLogger("everycoin.mlops.fetch")


# ── CoinGecko — prices ────────────────────────────────────────────────────────

async def fetch_prices(client: httpx.AsyncClient) -> None:
    """Fetch current price, 24h change, and market cap for all tracked coins."""
    log.info("Fetching prices for %d coins ...", len(COINS))
    try:
        res = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": ",".join(COINS),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        for coin_id in COINS:
            coin = data.get(coin_id, {})
            db.insert_price(
                coin_id=coin_id,
                price_usd=coin.get("usd"),
                change_24h=round(coin.get("usd_24h_change", 0) or 0, 4),
                market_cap=coin.get("usd_market_cap"),
            )
        log.info("  ✓ prices saved for %d coins", len(COINS))
    except Exception as e:
        log.error("  ✗ fetch_prices failed: %s", e)


# ── Etherscan — gas ───────────────────────────────────────────────────────────

async def fetch_gas(client: httpx.AsyncClient) -> None:
    """Fetch current Ethereum gas prices."""
    log.info("Fetching gas prices ...")
    try:
        res = await client.get(
            "https://api.etherscan.io/api",
            params={
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": ETHERSCAN_API_KEY,
            },
            timeout=10,
        )
        res.raise_for_status()
        body = res.json()
        if body.get("status") != "1":
            log.warning("  ✗ Etherscan gas returned status=%s — API key missing or rate limited", body.get("status"))
            return
        result = body.get("result", {})
        db.insert_gas(
            slow=_to_float(result.get("SafeGasPrice")),
            standard=_to_float(result.get("ProposeGasPrice")),
            fast=_to_float(result.get("FastGasPrice")),
        )
        log.info("  ✓ gas saved")
    except Exception as e:
        log.error("  ✗ fetch_gas failed: %s", e)


# ── DeFiLlama — TVL ───────────────────────────────────────────────────────────

async def fetch_defi(client: httpx.AsyncClient) -> None:
    """Fetch TVL for all tracked DeFi protocols."""
    log.info("Fetching DeFi TVL for %d protocols ...", len(PROTOCOLS))
    for protocol in PROTOCOLS:
        try:
            res = await client.get(
                f"https://api.llama.fi/protocol/{protocol}",
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            # tvl is a time-series list — take the latest entry
            tvl_series = data.get("tvl") or []
            latest_tvl = tvl_series[-1].get("totalLiquidityUSD") if tvl_series else None

            # chains come from currentChainTvls keys (strip suffixes like -staking)
            chain_tvls: dict = data.get("currentChainTvls") or {}
            chains = list({k.split("-")[0] for k in chain_tvls.keys()})[:10]

            db.insert_defi(
                protocol=protocol,
                tvl_usd=latest_tvl,
                category=data.get("category"),
                chains=chains,
            )
        except Exception as e:
            log.error("  ✗ fetch_defi failed for %s: %s", protocol, e)
    log.info("  ✓ DeFi TVL saved for %d protocols", len(PROTOCOLS))


# ── Main run — all fetchers in parallel ───────────────────────────────────────

async def run_pipeline() -> None:
    """Run all fetchers concurrently for one pipeline tick."""
    log.info("=== Pipeline tick started ===")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            fetch_prices(client),
            fetch_gas(client),
            fetch_defi(client),
        )
    counts = db.row_counts()
    log.info("=== Pipeline tick done | DB rows: %s ===", counts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None
