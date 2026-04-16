"""
Central config for the MLOps data pipeline.
Edit COINS to add/remove tokens being tracked.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── SQLite database path ──────────────────────────────────────────────────────
DB_PATH = ROOT / "mlops" / "data" / "market.db"

# ── Tokens to track (CoinGecko IDs) ──────────────────────────────────────────
COINS: list[str] = [
    "bitcoin",
    "ethereum",
    "solana",
    "binancecoin",
    "cardano",
    "avalanche-2",
    "chainlink",
    "uniswap",
]

# ── DeFi protocols to track (DeFiLlama slugs) ────────────────────────────────
PROTOCOLS: list[str] = [
    "aave",
    "uniswap",
    "lido",
    "curve-dex",
    "makerdao",
]

# ── Fetch interval in minutes ─────────────────────────────────────────────────
FETCH_INTERVAL_MINUTES: int = 60

# ── API keys ──────────────────────────────────────────────────────────────────
ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
