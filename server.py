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

import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("everycoin.server")

_http: httpx.AsyncClient  # shared client, initialized in lifespan

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


# ── /predict/btc-journey ─────────────────────────────────────────────────────

async def handle_btc_journey(request: Request) -> JSONResponse:
    """
    GET /predict/btc-journey?date=YYYY-MM-DD
    Returns hourly BTC prices + prediction for a 24h window.
    date defaults to the most recent prediction date.
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    from mlops import db as mlops_db
    from datetime import datetime, timedelta, timezone

    coin_id = "bitcoin"
    date_str = request.query_params.get("date")

    try:
        available_dates = mlops_db.available_forecast_dates(coin_id, limit=7)

        # Resolve which prediction to show
        if date_str:
            pred = mlops_db.prediction_for_date(coin_id, date_str)
        else:
            pred = mlops_db.latest_prediction(coin_id)
            date_str = pred["predicted_at"][:10] if pred else datetime.now(timezone.utc).date().isoformat()

        prices = []
        window_start = None
        window_end = None

        if pred:
            ws = pred["predicted_at"]
            dt_pred = datetime.fromisoformat(ws if "+" in ws else ws + "+00:00")
            # Window is exactly 9pm → 9pm (prediction time to +24h)
            window_start = dt_pred.isoformat(timespec="seconds")
            window_end   = (dt_pred + timedelta(hours=24)).isoformat(timespec="seconds")

            rows = mlops_db.price_history_range(coin_id, window_start, window_end)
            prices = [{"time": r["fetched_at"], "price": float(r["price_usd"])} for r in rows if r["price_usd"]]

            # If no prices yet, seed with the prediction's current_price at start
            if not prices:
                prices = [{"time": ws, "price": float(pred["current_price"])}]

        return JSONResponse({
            "date": date_str,
            "window_start": window_start,
            "window_end": window_end,
            "forecast": pred,
            "prices": prices,
            "available_dates": available_dates,
        }, headers=_CORS)

    except Exception as e:
        log.exception("btc-journey failed")
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /predict/price-history ───────────────────────────────────────────────────

async def handle_price_history(request: Request) -> JSONResponse:
    """
    GET /predict/price-history?coin=bitcoin&limit=7
    Returns past predictions with actual prices and outcomes.
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    coin_id = request.query_params.get("coin", "bitcoin")
    limit   = min(int(request.query_params.get("limit", 7)), 30)

    try:
        from mlops import db as mlops_db
        rows = mlops_db.prediction_history(coin_id, limit)
        # Only return settled predictions (actual price recorded)
        settled = [r for r in rows if r.get("actual_price")]
        return JSONResponse(settled, headers=_CORS)
    except Exception as e:
        log.exception("price-history failed for %s", coin_id)
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /whale/signals ────────────────────────────────────────────────────────────

async def handle_whale_signals(request: Request) -> JSONResponse:
    """
    GET /whale/signals
    Returns:
      A) transactions  — large BTC whale moves via mempool.space (free, no key)
      B) netflow       — derived buy/sell pressure from tx directions
      C) futures       — funding rate, open interest, long/short ratio (Binance, free)
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import asyncio
    import time

    # Known BTC exchange hot wallet addresses (publicly documented)
    EXCHANGE_WALLETS: dict[str, str] = {
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb": "Binance",
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance",
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97": "Binance",
        "1FzWLkAahHooV3kzTgyx6qsswXJ6sCXkSR": "Coinbase",
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64": "Coinbase",
        "bc1qazcm763858nkj2dj986etajv6wquslv8uxwczt": "Coinbase",
        "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g": "Kraken",
        "3E1jkD73KQXUL5LR1JmqaCMEXFG7ZoeFiy": "Kraken",
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s": "Bitfinex",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rq8dw": "Bitfinex",
        "1LdRcdxfbSnmCYYNdeYpUnztiYzVfBEQeC": "OKX",
        "bc1q9d3xa5gg45q2j39szguun6myggwydzhpf592de": "OKX",
    }

    MIN_BTC = 13.0   # ~$1M at $77k — small whale threshold

    async def fetch_whale_txns():
        try:
            # Fetch recent confirmed transactions from mempool.space
            # /api/v1/transactions gives recent mempool txs; we use block txs for confirmed
            blocks_r = await _http.get("https://mempool.space/api/v1/blocks/tip/height", timeout=6)
            if blocks_r.status_code != 200:
                return []
            tip = int(blocks_r.text.strip())

            # Get current BTC price for USD value calculation
            price_r = await _http.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                timeout=6,
            )
            btc_price = 80000.0
            if price_r.status_code == 200:
                btc_price = price_r.json().get("bitcoin", {}).get("usd", 80000)

            whale_txns = []

            # Scan last 3 blocks
            for block_height in range(tip, tip - 3, -1):
                hash_r = await _http.get(f"https://mempool.space/api/block-height/{block_height}", timeout=6)
                if hash_r.status_code != 200:
                    continue
                block_hash = hash_r.text.strip()

                # Get block summary for timestamp
                blk_r = await _http.get(f"https://mempool.space/api/block/{block_hash}", timeout=6)
                block_ts = int(time.time())
                if blk_r.status_code == 200:
                    block_ts = blk_r.json().get("timestamp", block_ts)

                # Fetch first page of txs (25 largest usually in first page)
                txs_r = await _http.get(f"https://mempool.space/api/block/{block_hash}/txs/0", timeout=10)
                if txs_r.status_code != 200:
                    continue

                for tx in txs_r.json():
                    # Sum all outputs (skip OP_RETURN)
                    vouts = [v for v in tx.get("vout", []) if v.get("value", 0) > 0]
                    total_sat = sum(v["value"] for v in vouts)
                    total_btc = total_sat / 1e8
                    if total_btc < MIN_BTC:
                        continue

                    amount_usd = int(total_btc * btc_price)

                    # Classify by matching output addresses to known exchange wallets
                    out_addrs  = [v.get("scriptpubkey_address", "") for v in vouts]
                    to_exchange = next((EXCHANGE_WALLETS[a] for a in out_addrs if a in EXCHANGE_WALLETS), None)

                    # Input addresses (vin scriptpubkey_address if available)
                    in_addrs    = [v.get("prevout", {}).get("scriptpubkey_address", "") for v in tx.get("vin", [])]
                    from_exchange = next((EXCHANGE_WALLETS[a] for a in in_addrs if a in EXCHANGE_WALLETS), None)

                    if to_exchange:
                        signal     = "sell"   # moving TO exchange → likely selling
                        from_label = "Wallet"
                        to_label   = to_exchange
                    elif from_exchange:
                        signal     = "buy"    # moving FROM exchange → accumulation
                        from_label = from_exchange
                        to_label   = "Wallet"
                    else:
                        signal     = "move"   # wallet-to-wallet (OTC or cold storage)
                        from_label = "Wallet"
                        to_label   = "Wallet"

                    # Size tier
                    if amount_usd >= 50_000_000:
                        tier = "market mover"
                    elif amount_usd >= 10_000_000:
                        tier = "big whale"
                    else:
                        tier = "whale"

                    whale_txns.append({
                        "hash":        tx.get("txid", "")[:12],
                        "amount_btc":  round(total_btc, 1),
                        "amount_usd":  amount_usd,
                        "from_label":  from_label,
                        "to_label":    to_label,
                        "signal":      signal,
                        "tier":        tier,
                        "timestamp":   block_ts,
                    })

                    if len(whale_txns) >= 8:
                        break
                if len(whale_txns) >= 8:
                    break

            # Sort by size, return top 5
            whale_txns.sort(key=lambda x: x["amount_btc"], reverse=True)
            return whale_txns[:5]

        except Exception:
            log.exception("fetch_whale_txns failed")
            return []

    async def fetch_okx_futures():
        # OKX public endpoints — no key, no US geo-block
        try:
            funding_r, oi_r, lsr_r = await asyncio.gather(
                _http.get("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP", timeout=6),
                _http.get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP", timeout=6),
                _http.get("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader?instId=BTC-USDT-SWAP&period=1H", timeout=6),
                return_exceptions=True,
            )

            funding_rate = None
            if not isinstance(funding_r, Exception) and funding_r.status_code == 200:
                data = funding_r.json().get("data", [])
                if data:
                    funding_rate = round(float(data[0]["fundingRate"]) * 100, 4)  # as %

            open_interest_usd = None
            oi_trend = None
            if not isinstance(oi_r, Exception) and oi_r.status_code == 200:
                data = oi_r.json().get("data", [])
                if data:
                    open_interest_usd = float(data[0]["oiUsd"])

            long_short_ratio = None
            if not isinstance(lsr_r, Exception) and lsr_r.status_code == 200:
                rows = lsr_r.json().get("data", [])
                if len(rows) >= 2:
                    curr = float(rows[0][1])
                    prev = float(rows[1][1])
                    long_short_ratio = round(curr, 3)
                    oi_trend = "up" if curr > prev else "down"
                elif len(rows) == 1:
                    long_short_ratio = round(float(rows[0][1]), 3)

            return {
                "funding_rate":      funding_rate,
                "open_interest_usd": open_interest_usd,
                "long_short_ratio":  long_short_ratio,
                "oi_trend":          oi_trend,
            }
        except Exception:
            log.exception("fetch_okx_futures failed")
            return {}

    txns, futures = await asyncio.gather(fetch_whale_txns(), fetch_okx_futures())

    # Netflow signal derived from transaction directions
    sells = sum(1 for t in txns if t["signal"] == "sell")
    buys  = sum(1 for t in txns if t["signal"] == "buy")
    if sells > buys:
        netflow_signal = "sell_pressure"
    elif buys > sells:
        netflow_signal = "accumulation"
    else:
        netflow_signal = "neutral"

    return JSONResponse({
        "transactions":   txns,
        "netflow_signal": netflow_signal,
        "netflow_sells":  sells,
        "netflow_buys":   buys,
        "futures":        futures,
    }, headers=_CORS)


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
    global _http
    _http = httpx.AsyncClient(timeout=10)
    rag.init_rag()

    # Run MLOps scheduler in background thread
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
    await _http.aclose()


starlette_app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/api/chat", handle_chat, methods=["POST", "OPTIONS"]),
        Route("/mcp", handle_mcp, methods=["POST"]),
        Route("/prices", handle_prices, methods=["GET", "OPTIONS"]),
        Route("/predict/ai-score", handle_predict, methods=["GET"]),
        Route("/predict/price-target", handle_price_target, methods=["GET", "OPTIONS"]),
        Route("/predict/btc-journey", handle_btc_journey, methods=["GET", "OPTIONS"]),
        Route("/predict/price-history", handle_price_history, methods=["GET", "OPTIONS"]),
        Route("/whale/signals", handle_whale_signals, methods=["GET", "OPTIONS"]),
        Route("/health", handle_health, methods=["GET"]),
    ],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info("EveryCoin server starting on http://0.0.0.0:%d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
