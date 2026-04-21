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
        import httpx
        coin_ids = [cid.strip() for cid in ids.split(",") if cid.strip()]
        # Single batched request — avoids 4 separate CoinGecko calls that each
        # count against the free-tier rate limit (causing 429s on rapid refreshes).
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": ",".join(coin_ids),
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                },
                timeout=10,
            )
        log.info("CoinGecko batch → status=%s ids=%s", res.status_code, coin_ids)
        if res.status_code != 200:
            log.warning("CoinGecko batch error: %s", res.text[:300])
            return JSONResponse({}, headers=_CORS)
        data = res.json()
        results = {}
        for coin_id in coin_ids:
            row = data.get(coin_id)
            if row:
                results[coin_id] = {
                    "price_usd": row.get("usd"),
                    "change_24h_pct": round(row.get("usd_24h_change", 0), 2),
                }
        log.info("handle_prices returning %d coins: %s", len(results), list(results.keys()))
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


# ── /predict/price-target ────────────────────────────────────────────────────

async def handle_price_target(request: Request) -> JSONResponse:
    """
    GET /predict/price-target?coin=bitcoin
    Returns a 24h price target derived from current signals + model confidence.

    Logic:
      - Base expected move = return_1h × 24 (linear extrapolation)
      - Scaled by AI score conviction: score 50 = no move, score 100 = full move
      - Dampened by confidence: low confidence shrinks the predicted move
      - Capped at ±15% to stay realistic
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    coin_id = request.query_params.get("coin", "bitcoin")

    try:
        import httpx
        from mlops.serve import predict as ml_predict
        from mlops import db as mlops_db
        from datetime import datetime, timezone

        # ── Serve from daily DB prediction if it exists and is from today ──────
        stored = mlops_db.latest_prediction(coin_id)
        if stored:
            predicted_at = datetime.fromisoformat(stored["predicted_at"])
            age_hours = (datetime.now(timezone.utc) - predicted_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if age_hours <= 24:
                log.info("Serving stored daily prediction for %s (age=%.1fh)", coin_id, age_hours)
                return JSONResponse({
                    "coin_id":            coin_id,
                    "current_price":      stored["current_price"],
                    "predicted_price":    stored["predicted_price"],
                    "predicted_move_pct": stored["predicted_move_pct"],
                    "predicted_score":    stored["predicted_score"],
                    "confidence":         stored["confidence"],
                    "generated_at":       stored["predicted_at"] + "Z",
                    "actual_price":       stored.get("actual_price"),
                    "outcome":            stored.get("outcome"),
                    "source":             "scheduled",
                }, headers=_CORS)

        # ── Fallback: compute live ────────────────────────────────────────────
        # Get AI signals
        pred = ml_predict(coin_id)
        if pred.get("error") or not pred.get("signal"):
            return JSONResponse({"error": "No signal data available"}, status_code=503, headers=_CORS)

        signal = pred["signal"]
        ai_score = pred["ai_score"]        # 0–100
        confidence = pred["confidence"]    # 0–1

        # return_1h is already in % (e.g. 0.3 means 0.3%)
        return_1h = signal.get("return_1h", 0)

        # Extrapolate 1h trend to 24h, dampened heavily (momentum fades)
        raw_move_pct = return_1h * 8  # 8h equivalent, not full 24 (momentum dampener)

        # Score conviction: score=50 → 0 bias, score=100 → full bullish, score=0 → full bearish
        conviction = (ai_score - 50) / 50  # -1 to +1

        # Blend: 60% momentum, 40% AI conviction
        blended_pct = (raw_move_pct * 0.6) + (conviction * abs(raw_move_pct) * 0.4 + conviction * 2)

        # Dampen by confidence (low confidence = smaller prediction)
        predicted_move_pct = blended_pct * confidence

        # Cap at ±15%
        predicted_move_pct = max(-15.0, min(15.0, predicted_move_pct))
        predicted_move_pct = round(predicted_move_pct, 2)

        # Fetch current BTC price
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=8,
            )
        if res.status_code != 200:
            return JSONResponse({"error": "Price fetch failed"}, status_code=503, headers=_CORS)

        current_price = res.json().get(coin_id, {}).get("usd")
        if not current_price:
            return JSONResponse({"error": "Price unavailable"}, status_code=503, headers=_CORS)

        predicted_price = round(current_price * (1 + predicted_move_pct / 100), 2)
        predicted_score = min(100, max(0, round(ai_score + conviction * 10 * confidence)))

        log.info(
            "price-target %s: current=$%.2f move=%.2f%% target=$%.2f predicted_score=%d",
            coin_id, current_price, predicted_move_pct, predicted_price, predicted_score,
        )

        return JSONResponse({
            "coin_id": coin_id,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_move_pct": predicted_move_pct,
            "predicted_score": predicted_score,
            "confidence": pred["confidence"],
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }, headers=_CORS)

    except Exception as e:
        log.exception("price-target failed for %s", coin_id)
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
        Route("/predict/price-target", handle_price_target, methods=["GET", "OPTIONS"]),
        Route("/health", handle_health, methods=["GET"]),
    ],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info("EveryCoin server starting on http://0.0.0.0:%d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
