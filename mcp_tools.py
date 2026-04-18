"""
MCP Tool definitions + implementations.
Tools are defined in MCP format and also exposed as plain async functions
so LangGraph nodes can call them directly.
"""

import json
import logging
import os

import httpx
from mcp import types

import rag
from mlops.serve import predict as ml_predict

log = logging.getLogger("everycoin.tools")


# ── MCP tool registry (served via POST /mcp) ──────────────────────────────────

MCP_TOOLS: list[types.Tool] = [
    types.Tool(
        name="get_token_price",
        description="Get current USD price, 24h change, and market cap for a crypto token",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "CoinGecko coin id e.g. bitcoin, ethereum, solana",
                }
            },
            "required": ["symbol"],
        },
    ),
    types.Tool(
        name="get_defi_stats",
        description="Get TVL, category, chains, and description for a DeFi protocol",
        inputSchema={
            "type": "object",
            "properties": {
                "protocol": {
                    "type": "string",
                    "description": "Protocol slug e.g. aave, uniswap, curve, lido, gmx",
                }
            },
            "required": ["protocol"],
        },
    ),
    types.Tool(
        name="get_gas_price",
        description="Get current Ethereum gas prices (slow, standard, fast) in Gwei",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="analyze_wallet",
        description="Analyze an Ethereum wallet — native balance, token holdings, recent transactions",
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Ethereum wallet address starting with 0x",
                }
            },
            "required": ["address"],
        },
    ),
    types.Tool(
        name="predict_ai_score",
        description="Get ML-predicted AI score (0-100), price direction (BUY/HOLD/SELL), and confidence for a crypto coin based on technical indicators",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "CoinGecko coin id e.g. bitcoin, ethereum, solana",
                }
            },
            "required": ["symbol"],
        },
    ),
    types.Tool(
        name="search_knowledge",
        description="Semantic search over EveryCoin crypto knowledge base — DeFi, security, strategy, L2s",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query e.g. 'how does uniswap v3 work'",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic filter: defi, security, strategy, l2",
                    "enum": ["defi", "security", "strategy", "l2"],
                },
            },
            "required": ["query"],
        },
    ),
]


# ── Tool implementations ───────────────────────────────────────────────────────

async def get_token_price(symbol: str) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": symbol.lower(),
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
                timeout=10,
            )
            data = res.json()
            coin_id = symbol.lower()
            if coin_id not in data:
                return {"error": f"Token '{symbol}' not found on CoinGecko."}
            coin = data[coin_id]
            return {
                "symbol": coin_id,
                "price_usd": coin.get("usd"),
                "change_24h_pct": round(coin.get("usd_24h_change", 0), 2),
                "market_cap_usd": coin.get("usd_market_cap"),
            }
        except Exception as e:
            return {"error": str(e)}


async def get_defi_stats(protocol: str) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                f"https://api.llama.fi/protocol/{protocol.lower()}",
                timeout=10,
            )
            data = res.json()
            return {
                "name": data.get("name"),
                "tvl_usd": data.get("tvl"),
                "category": data.get("category"),
                "chains": data.get("chains", [])[:5],
                "description": (data.get("description") or "")[:300],
            }
        except Exception as e:
            return {"error": str(e)}


async def get_gas_price() -> dict:
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "gastracker",
                    "action": "gasoracle",
                    "apikey": api_key,
                },
                timeout=10,
            )
            data = res.json().get("result", {})
            return {
                "slow_gwei": data.get("SafeGasPrice"),
                "standard_gwei": data.get("ProposeGasPrice"),
                "fast_gwei": data.get("FastGasPrice"),
            }
        except Exception as e:
            return {"error": str(e)}


async def analyze_wallet(address: str) -> dict:
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    base = "https://api.etherscan.io/api"
    async with httpx.AsyncClient() as client:
        try:
            # Native ETH balance
            bal_res = await client.get(base, params={
                "module": "account", "action": "balance",
                "address": address, "tag": "latest", "apikey": api_key,
            }, timeout=10)
            balance_wei = int(bal_res.json().get("result", 0))
            balance_eth = round(balance_wei / 1e18, 6)

            # Recent transactions (last 5)
            tx_res = await client.get(base, params={
                "module": "account", "action": "txlist",
                "address": address, "startblock": 0, "endblock": 99999999,
                "page": 1, "offset": 5, "sort": "desc", "apikey": api_key,
            }, timeout=10)
            txs = tx_res.json().get("result", [])
            recent_txs = []
            if isinstance(txs, list):
                for tx in txs[:5]:
                    recent_txs.append({
                        "hash": tx.get("hash", "")[:12] + "...",
                        "from": tx.get("from", "")[:10] + "...",
                        "to": (tx.get("to") or "contract")[:10] + "...",
                        "value_eth": round(int(tx.get("value", 0)) / 1e18, 6),
                        "age_blocks": tx.get("confirmations"),
                    })

            return {
                "address": address,
                "balance_eth": balance_eth,
                "recent_transactions": recent_txs,
            }
        except Exception as e:
            return {"error": str(e)}


def predict_ai_score(symbol: str) -> dict:
    try:
        return ml_predict(symbol.lower())
    except Exception as e:
        return {"error": str(e), "coin_id": symbol, "ai_score": 50, "direction": "HOLD"}


def search_knowledge(query: str, topic: str | None = None) -> dict:
    results = rag.search(query, topic=topic, n_results=3)
    if not results:
        return {"results": [], "message": "No relevant knowledge found."}
    return {
        "results": [
            {"text": r["text"], "source": r["source"], "relevance": 1 - r["distance"]}
            for r in results
        ]
    }


# ── Dispatcher — called by MCP handler and LangGraph nodes ────────────────────

async def call_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return result as JSON string."""
    log.info("  🔧 MCP tool called : %s | args: %s", name, arguments)

    if name == "predict_ai_score":
        result = predict_ai_score(arguments["symbol"])
    elif name == "get_token_price":
        result = await get_token_price(arguments["symbol"])
    elif name == "get_defi_stats":
        result = await get_defi_stats(arguments["protocol"])
    elif name == "get_gas_price":
        result = await get_gas_price()
    elif name == "analyze_wallet":
        result = await analyze_wallet(arguments["address"])
    elif name == "search_knowledge":
        result = search_knowledge(arguments["query"], arguments.get("topic"))
    else:
        result = {"error": f"Unknown tool: {name}"}

    status = "✓" if "error" not in result else "✗"
    log.info("  %s MCP tool result : %s | %s", status, name, str(result)[:120])
    return json.dumps(result)
