"""
EveryCoin MCP Server — HTTP transport
Demonstrates agentic AI tool use via the Model Context Protocol.

Start locally:  python server.py
Railway:        auto-starts via Procfile
"""

import asyncio
import json
import logging
import os

import httpx
import uvicorn
from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("everycoin-mcp")

app = Server("everycoin-mcp")


# ── Tool registry ──────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    log.info("tools/list called")
    return [
        types.Tool(
            name="get_token_price",
            description="Get current USD price and 24h change for a crypto token",
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
            name="get_defi_protocol_stats",
            description="Get TVL, category and chain info for a DeFi protocol",
            inputSchema={
                "type": "object",
                "properties": {
                    "protocol": {
                        "type": "string",
                        "description": "Protocol slug e.g. aave, uniswap, curve, lido",
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
            name="search_crypto_knowledge",
            description="Search internal knowledge base for DeFi/crypto explanations",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "e.g. 'how does uniswap v3 work'",
                    }
                },
                "required": ["query"],
            },
        ),
    ]


# ── Tool execution ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    log.info("tools/call → %s %s", name, arguments)

    if name == "get_token_price":
        return await _get_token_price(arguments["symbol"])
    if name == "get_defi_protocol_stats":
        return await _get_defi_stats(arguments["protocol"])
    if name == "get_gas_price":
        return await _get_gas_price()
    if name == "search_crypto_knowledge":
        return _search_knowledge(arguments["query"])

    raise ValueError(f"Unknown tool: {name}")


# ── Tool implementations ───────────────────────────────────────────────────────

async def _get_token_price(symbol: str) -> list[types.TextContent]:
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
                return [types.TextContent(type="text", text=f"Token '{symbol}' not found.")]
            coin = data[coin_id]
            result = {
                "symbol": coin_id,
                "price_usd": coin.get("usd"),
                "change_24h_pct": round(coin.get("usd_24h_change", 0), 2),
                "market_cap_usd": coin.get("usd_market_cap"),
            }
            return [types.TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]


async def _get_defi_stats(protocol: str) -> list[types.TextContent]:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                f"https://api.llama.fi/protocol/{protocol.lower()}",
                timeout=10,
            )
            data = res.json()
            result = {
                "name": data.get("name"),
                "tvl_usd": data.get("tvl"),
                "category": data.get("category"),
                "chains": data.get("chains", [])[:5],
                "description": (data.get("description") or "")[:300],
            }
            return [types.TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]


async def _get_gas_price() -> list[types.TextContent]:
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
            result = {
                "slow_gwei": data.get("SafeGasPrice"),
                "standard_gwei": data.get("ProposeGasPrice"),
                "fast_gwei": data.get("FastGasPrice"),
            }
            return [types.TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]


def _search_knowledge(query: str) -> list[types.TextContent]:
    knowledge = [
        {
            "keywords": ["uniswap", "amm", "concentrated liquidity", "v3"],
            "text": "Uniswap V3 uses concentrated liquidity — LPs set custom price ranges, up to 4000x capital efficiency vs V2. Higher fee earnings but increased impermanent loss risk outside range.",
        },
        {
            "keywords": ["aave", "lending", "liquidation", "health factor"],
            "text": "Aave is a decentralised lending protocol. Health Factor <1.0 triggers liquidation. Flash loans allow uncollateralised borrows within one tx.",
        },
        {
            "keywords": ["rug pull", "scam", "honeypot", "red flag"],
            "text": "Rug pull red flags: anonymous team, no audit, mint functions, locked liquidity <6 months, sell tax >10%. Tools: Token Sniffer, Honeypot.is, DEXTools.",
        },
        {
            "keywords": ["l2", "layer 2", "arbitrum", "optimism", "base", "zksync"],
            "text": "Optimistic rollups (Arbitrum, Optimism, Base) use fraud proofs with ~7 day withdrawal. ZK rollups (zkSync, Starknet) use cryptographic proofs — faster finality.",
        },
        {
            "keywords": ["mev", "maximal extractable value", "frontrun", "sandwich"],
            "text": "MEV: bots reorder txs for profit. Sandwich attacks buy before/sell after your DEX trade. Protection: Flashbots Protect RPC, low slippage, CoW Swap.",
        },
        {
            "keywords": ["yield farming", "liquidity mining", "apr", "apy"],
            "text": "Yield farming risks: smart contract exploits, impermanent loss, token inflation. APY >100% is usually unsustainable. Always check audits and emission schedules.",
        },
        {
            "keywords": ["cold wallet", "hardware wallet", "ledger", "trezor", "seed"],
            "text": "Cold wallet best practices: buy from manufacturer, never enter seed digitally, store on metal, use passphrase. Ledger and Trezor are the leading options.",
        },
        {
            "keywords": ["bear market", "strategy", "dca", "accumulate"],
            "text": "Bear market strategy: DCA high-conviction assets, reduce altcoin exposure, hold stablecoins, avoid leverage. Crypto bear markets typically last 12-18 months.",
        },
    ]

    q = query.lower()
    matches = [k["text"] for k in knowledge if any(kw in q for kw in k["keywords"])]

    if not matches:
        return [types.TextContent(type="text", text="No specific knowledge entry found.")]

    return [types.TextContent(type="text", text="\n\n".join(matches))]


# ── HTTP handler (JSON-RPC over POST /mcp) ─────────────────────────────────────

async def handle_mcp(request: Request) -> JSONResponse:
    body = await request.json()
    method = body.get("method")
    req_id = body.get("id", 1)
    params = body.get("params", {})

    try:
        if method == "tools/list":
            tools = await list_tools()
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema,
                        }
                        for t in tools
                    ]
                },
            })

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            content = await call_tool(name, arguments)
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": c.type, "text": c.text} for c in content]
                },
            })

        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}},
            status_code=404,
        )

    except Exception as e:
        log.exception("Error handling %s", method)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}},
            status_code=500,
        )


async def handle_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "everycoin-mcp", "tools": 4})


# ── Starlette app ──────────────────────────────────────────────────────────────

starlette_app = Starlette(
    routes=[
        Route("/mcp", handle_mcp, methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
    ]
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info("EveryCoin MCP server starting on http://0.0.0.0:%d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
