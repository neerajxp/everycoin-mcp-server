"""
EveryCoin HTTP Server — thin layer only.

Routes:
  POST /api/chat  — LangGraph multi-agent pipeline
  POST /mcp       — MCP JSON-RPC (tools/list, tools/call)
  GET  /health    — Health check

Start locally:  python server.py
Railway:        auto-starts via Procfile
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import threading

import mcp_tools
import rag
from graph import run_graph
from mlops.serve import predict, predict_all

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("everycoin.server")

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
}


# ── /api/chat ─────────────────────────────────────────────────────────────────

async def handle_chat(request: Request) -> JSONResponse:
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400, headers=_CORS)

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            {"error": "Invalid request: messages array required."},
            status_code=400,
            headers=_CORS,
        )

    # Extract last user message as the current query
    user_query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if not user_query:
        return JSONResponse({"error": "No user message found."}, status_code=400, headers=_CORS)

    session_id = body.get("session_id")
    user_id = body.get("user_id", "anonymous")

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY is not set")
        return JSONResponse({"error": "Server configuration error"}, status_code=500, headers=_CORS)

    try:
        answer = await run_graph(
            user_query=user_query,
            messages=messages,
            session_id=session_id,
            user_id=user_id,
        )
        return JSONResponse({"content": answer}, headers=_CORS)
    except Exception as e:
        log.exception("Graph execution failed")
        return JSONResponse({"error": str(e)}, status_code=502, headers=_CORS)


# ── /mcp  (MCP JSON-RPC) ──────────────────────────────────────────────────────

async def handle_mcp(request: Request) -> JSONResponse:
    body = await request.json()
    method = body.get("method")
    req_id = body.get("id", 1)
    params = body.get("params", {})

    try:
        if method == "tools/list":
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
                        for t in mcp_tools.MCP_TOOLS
                    ]
                },
            })

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            result_text = await mcp_tools.call_tool(name, arguments)
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}]
                },
            })

        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id,
             "error": {"code": -32601, "message": f"Method not found: {method}"}},
            status_code=404,
        )

    except Exception as e:
        log.exception("MCP error handling %s", method)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id,
             "error": {"code": -32603, "message": str(e)}},
            status_code=500,
        )


# ── /prices ──────────────────────────────────────────────────────────────────

async def handle_prices(request: Request) -> JSONResponse:
    """
    GET /prices?ids=bitcoin,ethereum,solana
    Returns { coin_id: { price_usd, change_24h_pct } }
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    ids = request.query_params.get("ids", "")
    if not ids:
        return JSONResponse({"error": "ids param required"}, status_code=400, headers=_CORS)
    try:
        results = {}
        import asyncio
        tasks = [mcp_tools.get_token_price(cid.strip()) for cid in ids.split(",") if cid.strip()]
        fetched = await asyncio.gather(*tasks)
        for r in fetched:
            if "error" not in r:
                results[r["symbol"]] = {
                    "price_usd": r["price_usd"],
                    "change_24h_pct": r["change_24h_pct"],
                }
        return JSONResponse(results, headers=_CORS)
    except Exception as e:
        log.exception("Prices fetch failed")
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /predict/ai-score ────────────────────────────────────────────────────────

async def handle_predict(request: Request) -> JSONResponse:
    """
    GET  /predict/ai-score?coin=bitcoin   — single coin
    GET  /predict/ai-score                — all tracked coins
    """
    coin_id = request.query_params.get("coin")
    try:
        result = predict(coin_id) if coin_id else predict_all()
        return JSONResponse(result, headers=_CORS)
    except Exception as e:
        log.exception("Prediction failed for coin=%s", coin_id)
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /health ───────────────────────────────────────────────────────────────────

async def handle_health(_request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "server": "everycoin-mcp",
        "agents": ["router", "market_analyst", "defi_researcher",
                   "wallet_forensics", "knowledge_expert", "strategist"],
        "tools": [t.name for t in mcp_tools.MCP_TOOLS],
        "rag": "chromadb",
        "memory": "short-term + long-term",
    }, headers=_CORS)


# ── Starlette app ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app):
    rag.init_rag()

    # Run MLOps scheduler in background thread (same container = shared SQLite)
    def _run_scheduler():
        from mlops import db as mlops_db
        from mlops.scheduler import main as scheduler_main
        import sys
        sys.argv = ["scheduler"]  # clear uvicorn args
        mlops_db.init_db()
        scheduler_main()

    t = threading.Thread(target=_run_scheduler, daemon=True, name="mlops-scheduler")
    t.start()
    log.info("MLOps scheduler started in background thread")

    yield


starlette_app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/api/chat", handle_chat, methods=["POST", "OPTIONS"]),
        Route("/mcp", handle_mcp, methods=["POST"]),
        Route("/prices", handle_prices, methods=["GET", "OPTIONS"]),
        Route("/predict/ai-score", handle_predict, methods=["GET"]),
        Route("/health", handle_health, methods=["GET"]),
    ],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info("EveryCoin server starting on http://0.0.0.0:%d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
