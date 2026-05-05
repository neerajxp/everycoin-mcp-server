"""
EveryCoin HTTP Server — thin layer only.

Routes:
  POST /api/chat  — LangGraph multi-agent pipeline
  POST /mcp       — MCP JSON-RPC (tools/list, tools/call)
  GET  /health    — Health check

Start locally:  python server.py
Railway:        auto-starts via Procfile
"""

import json
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

# ── Shared caches — avoid duplicate external API calls ────────────────────────
import time as _time_cache

# CoinGecko simple/price cache — keyed by frozenset of coin IDs
# Saves ~3 redundant CoinGecko calls per user session
_prices_cache: dict = {}          # {frozenset(ids): {"data": {...}, "ts": float}}
_PRICES_TTL = 5 * 60              # 5 minutes

# CoinGecko BTC hourly chart — used by momentum + price-target
_btc_chart_cache: dict = {}       # {"data": [...], "ts": float}
_BTC_CHART_TTL = 30 * 60          # 30 minutes (matches momentum TTL)

# OKX futures data — funding rate, OI, L/S ratio
# Shared between /whale/signals and /predict/btc-momentum
_okx_cache: dict = {}             # {"data": {...}, "ts": float}
_OKX_TTL = 5 * 60                 # 5 minutes

# mempool.space whale transactions
_whale_txns_cache: dict = {}      # {"data": [...], "netflow": str, "ts": float}
_WHALE_TXNS_TTL = 3 * 60          # 3 minutes — new blocks every ~10 min, no need to hammer


async def _get_prices(coin_ids: list[str], force: bool = False) -> dict:
    """Cached CoinGecko simple/price fetch. Shared across all endpoints.
    Pass force=True to bypass cache (e.g. on user-initiated hard refresh)."""
    key = frozenset(coin_ids)
    now = _time_cache.time()
    cached = _prices_cache.get(key)
    if not force and cached and (now - cached["ts"]) < _PRICES_TTL:
        return cached["data"]

    try:
        res = await _http.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(coin_ids), "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()
            _prices_cache[key] = {"data": data, "ts": now}
            return data
    except Exception:
        log.warning("_get_prices failed for %s", coin_ids)
    return cached["data"] if cached else {}


async def _get_btc_chart() -> list[float]:
    """Cached CoinGecko hourly BTC price chart (48h). Shared across endpoints."""
    now = _time_cache.time()
    if _btc_chart_cache.get("ts") and (now - _btc_chart_cache["ts"]) < _BTC_CHART_TTL:
        return _btc_chart_cache["data"]

    try:
        res = await _http.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            "?vs_currency=usd&days=2&interval=hourly",
            timeout=10,
        )
        if res.status_code == 200:
            pts = res.json().get("prices", [])
            prices = [p[1] for p in pts]
            if prices:
                _btc_chart_cache["data"] = prices
                _btc_chart_cache["ts"] = now
                return prices
    except Exception:
        log.warning("_get_btc_chart failed")
    return _btc_chart_cache.get("data", [])


async def _get_okx_futures() -> dict:
    """Cached OKX futures data. Shared between /whale/signals and /predict/btc-momentum."""
    import asyncio
    now = _time_cache.time()
    if _okx_cache.get("ts") and (now - _okx_cache["ts"]) < _OKX_TTL:
        return _okx_cache["data"]

    try:
        funding_r, oi_r, lsr_r = await asyncio.gather(
            _http.get("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP", timeout=6),
            _http.get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP", timeout=6),
            _http.get("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader?instId=BTC-USDT-SWAP&period=1H", timeout=6),
            return_exceptions=True,
        )

        funding_rate = None
        if not isinstance(funding_r, Exception) and funding_r.status_code == 200:
            d = funding_r.json().get("data", [])
            if d:
                funding_rate = round(float(d[0]["fundingRate"]) * 100, 4)

        open_interest_usd = None
        oi_trend = None
        if not isinstance(oi_r, Exception) and oi_r.status_code == 200:
            d = oi_r.json().get("data", [])
            if d:
                open_interest_usd = float(d[0]["oiUsd"])

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

        result = {
            "funding_rate":      funding_rate,
            "open_interest_usd": open_interest_usd,
            "long_short_ratio":  long_short_ratio,
            "oi_trend":          oi_trend,
        }
        _okx_cache["data"] = result
        _okx_cache["ts"] = now
        return result
    except Exception:
        log.warning("_get_okx_futures failed")
        return _okx_cache.get("data", {})


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

def _is_hard_refresh(request: Request) -> bool:
    """True when the browser signals Cache-Control: no-cache (hard refresh)."""
    return request.headers.get("cache-control", "").lower() == "no-cache"


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
        coin_ids = [cid.strip() for cid in ids.split(",") if cid.strip()]
        # Bypass cache on hard refresh so the user always sees fresh prices
        data = await _get_prices(coin_ids, force=_is_hard_refresh(request))
        results = {}
        for coin_id in coin_ids:
            row = data.get(coin_id)
            if row:
                results[coin_id] = {
                    "price_usd": row.get("usd"),
                    "change_24h_pct": round(row.get("usd_24h_change", 0), 2),
                }
        log.info("handle_prices returning %d coins (cached=%s)", len(results), bool(data))
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

        # Fetch current price via shared cache
        price_data = await _get_prices([coin_id])
        current_price = price_data.get(coin_id, {}).get("usd")
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


# ── Technical indicator helpers ───────────────────────────────────────────────

def _ema(prices: list, period: int) -> list:
    k = 2 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _compute_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
    gains = sum(d for d in deltas if d > 0)
    losses = sum(-d for d in deltas if d < 0)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_macd(prices: list) -> tuple:
    """Returns (macd_line, signal_line, histogram) for latest candle."""
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(prices))]
    signal = _ema(macd_line[-9:], 9) if len(macd_line) >= 9 else [macd_line[-1]]
    hist = macd_line[-1] - signal[-1]
    return macd_line[-1], signal[-1], hist


def _compute_atr(prices: list, period: int = 14) -> float:
    """Approximated ATR using |close[i] - close[i-1]| (no OHLC available)."""
    if len(prices) < period + 1:
        return 0.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(len(prices) - period, len(prices))]
    return sum(trs) / period


# ── /predict/btc-momentum in-memory cache ─────────────────────────────────────

_momentum_cache: dict = {}          # {"data": ..., "ts": float}
_MOMENTUM_TTL = 30 * 60             # 30 minutes


# ── /predict/btc-momentum ────────────────────────────────────────────────────

async def handle_btc_momentum(request: Request) -> JSONResponse:
    """
    GET /predict/btc-momentum
    Option C: blends the trained ML score (50%) with live indicator score (50%).
    Picks a dynamic target window based on signal strength.
    Cached 30 min in memory.

    Response shape:
      {
        "blended_score":    0-100,
        "ml_score":         0-100,
        "live_score":       0-100,
        "direction":        "BUY"|"HOLD"|"SELL",
        "confidence":       0.0-1.0,
        "current_price":    float,
        "target_price":     float,
        "target_pct":       float,          # e.g. 2.3 or -1.1
        "window_hours":     int,            # 1 | 4 | 6 | 12 | 24
        "forecast_at":      ISO str,
        "target_at":        ISO str,
        "signals": {
            "rsi":          float,
            "macd":         float,
            "atr":          float,
            "return_1h":    float,
            "return_4h":    float,
            "funding_rate": float | null,
            "oi_trend":     "up"|"down"|null,
            "whale_flow":   "accumulation"|"sell_pressure"|"neutral"
        },
        "cached": bool
      }
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import time as _time
    import asyncio
    from datetime import datetime, timezone, timedelta

    now_ts = _time.time()

    # Serve from cache if still fresh (skip on hard refresh)
    if not _is_hard_refresh(request) and _momentum_cache.get("ts") and (now_ts - _momentum_cache["ts"]) < _MOMENTUM_TTL:
        data = dict(_momentum_cache["data"])
        data["cached"] = True
        return JSONResponse(data, headers=_CORS)

    try:
        # ── 1. Fetch BTC prices + OKX data — both use shared caches ──────────────
        prices_raw, okx_data = await asyncio.gather(
            _get_btc_chart(),
            _get_okx_futures(),
        )

        current_price: float = 0.0
        if prices_raw:
            current_price = prices_raw[-1]

        if not prices_raw or current_price == 0:
            return JSONResponse({"error": "Unable to fetch price data"}, status_code=503, headers=_CORS)

        # ── 2. Compute live indicators ─────────────────────────────────────────
        rsi   = _compute_rsi(prices_raw, 14)
        macd, _, macd_hist = _compute_macd(prices_raw)
        atr   = _compute_atr(prices_raw, 14)

        ret_1h  = ((prices_raw[-1] / prices_raw[-2])  - 1) * 100 if len(prices_raw) >= 2  else 0.0
        ret_4h  = ((prices_raw[-1] / prices_raw[-5])  - 1) * 100 if len(prices_raw) >= 5  else 0.0
        ret_24h = ((prices_raw[-1] / prices_raw[-25]) - 1) * 100 if len(prices_raw) >= 25 else 0.0

        # Funding rate from shared OKX cache
        funding_rate: float | None = okx_data.get("funding_rate")

        # Whale netflow — use in-process cache to avoid HTTP self-call
        whale_flow = "neutral"
        if _whale_txns_cache.get("ts") and (_time_cache.time() - _whale_txns_cache["ts"]) < _WHALE_TXNS_TTL:
            whale_flow = _whale_txns_cache.get("netflow", "neutral")

        # ── 3. Live score (0–100) ──────────────────────────────────────────────
        # Each signal contributes to a raw score, then normalise to 0–100.
        live_components: list[float] = []

        # RSI: 0 = deeply oversold (30→bullish reversal), 100 = deeply overbought
        # We score it so >70 = overbought (bearish), <30 = oversold (bullish)
        if rsi <= 30:
            live_components.append(80.0)   # strong buy signal
        elif rsi <= 45:
            live_components.append(62.0)
        elif rsi <= 55:
            live_components.append(50.0)
        elif rsi <= 70:
            live_components.append(40.0)
        else:
            live_components.append(20.0)   # overbought → bearish

        # MACD histogram
        if macd_hist > 0:
            live_components.append(min(50 + macd_hist / atr * 30, 85) if atr > 0 else 65)
        else:
            live_components.append(max(50 + macd_hist / atr * 30, 15) if atr > 0 else 35)

        # 4h return
        if ret_4h > 1.5:
            live_components.append(75.0)
        elif ret_4h > 0.3:
            live_components.append(60.0)
        elif ret_4h > -0.3:
            live_components.append(50.0)
        elif ret_4h > -1.5:
            live_components.append(38.0)
        else:
            live_components.append(22.0)

        # Funding rate (positive = longs paying → slightly bearish sentiment)
        if funding_rate is not None:
            if funding_rate < -0.03:
                live_components.append(72.0)   # shorts paying → bullish
            elif funding_rate < 0.03:
                live_components.append(52.0)
            elif funding_rate < 0.08:
                live_components.append(42.0)
            else:
                live_components.append(25.0)   # over-leveraged longs → bearish

        # Whale flow
        if whale_flow == "accumulation":
            live_components.append(72.0)
        elif whale_flow == "sell_pressure":
            live_components.append(28.0)
        else:
            live_components.append(50.0)

        live_score = round(sum(live_components) / len(live_components))

        # ── 4. ML score (from trained model) ──────────────────────────────────
        ml_score = 50
        ml_confidence = 0.5
        try:
            ml_result = predict("bitcoin")
            if not ml_result.get("error"):
                ml_score = ml_result["ai_score"]
                ml_confidence = ml_result["confidence"]
        except Exception:
            pass

        # ── 5. Blended score: ML 50% + live 50% ───────────────────────────────
        blended = round(ml_score * 0.5 + live_score * 0.5)

        # ── 6. Direction from blended score ───────────────────────────────────
        if blended >= 62:
            direction = "BUY"
        elif blended <= 38:
            direction = "SELL"
        else:
            direction = "HOLD"

        # ── 7. Dynamic window based on signal agreement & momentum strength ────
        # Signal agreement: do ML and live agree on direction?
        ml_bull  = ml_score >= 55
        live_bull = live_score >= 55
        ml_bear  = ml_score <= 45
        live_bear = live_score <= 45
        signals_agree = (ml_bull and live_bull) or (ml_bear and live_bear)

        abs_dev = abs(blended - 50)   # 0 = totally neutral, 50 = max conviction

        if abs_dev >= 20 and signals_agree and abs(ret_4h) > 1.0:
            window_hours = 4
        elif abs_dev >= 15 and signals_agree:
            window_hours = 6
        elif abs_dev >= 10:
            window_hours = 12
        else:
            window_hours = 24

        # ── 8. Target price via ATR × window multiplier ────────────────────────
        atr_hourly = atr  # already hourly ATR
        multiplier = {4: 1.5, 6: 2.0, 12: 2.5, 24: 3.0}[window_hours]
        raw_move = atr_hourly * multiplier

        # Clamp to realistic range: 0.2% – 5%
        max_move = current_price * 0.05
        min_move = current_price * 0.002
        move = max(min_move, min(raw_move, max_move))

        if direction == "BUY":
            target_price = round(current_price + move, 0)
        elif direction == "SELL":
            target_price = round(current_price - move, 0)
        else:
            # HOLD — narrow band, direction driven by blended lean
            lean = 1 if blended >= 50 else -1
            target_price = round(current_price + lean * move * 0.5, 0)

        target_pct = round((target_price / current_price - 1) * 100, 2)

        # ── 9. Confidence: combination of model confidence + signal agreement ──
        agreement_bonus = 0.08 if signals_agree else -0.05
        confidence = round(min(1.0, max(0.1, ml_confidence + agreement_bonus)), 3)

        # ── 10. Timestamps ────────────────────────────────────────────────────
        now_dt = datetime.now(timezone.utc)
        forecast_at = now_dt.isoformat(timespec="seconds")
        target_at   = (now_dt + timedelta(hours=window_hours)).isoformat(timespec="seconds")

        result = {
            "blended_score":  blended,
            "ml_score":       ml_score,
            "live_score":     live_score,
            "direction":      direction,
            "confidence":     confidence,
            "current_price":  round(current_price, 0),
            "target_price":   target_price,
            "target_pct":     target_pct,
            "window_hours":   window_hours,
            "forecast_at":    forecast_at,
            "target_at":      target_at,
            "signals": {
                "rsi":          round(rsi, 1),
                "macd":         round(macd, 2),
                "macd_hist":    round(macd_hist, 2),
                "atr":          round(atr, 0),
                "return_1h":    round(ret_1h, 2),
                "return_4h":    round(ret_4h, 2),
                "return_24h":   round(ret_24h, 2),
                "funding_rate": funding_rate,
                "whale_flow":   whale_flow,
            },
            "cached": False,
        }

        _momentum_cache["data"] = result
        _momentum_cache["ts"]   = now_ts

        return JSONResponse(result, headers=_CORS)

    except Exception as e:
        log.exception("btc-momentum failed")
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

    # Serve from cache if fresh (skip on hard refresh)
    now_ts = time.time()
    if not _is_hard_refresh(request) and _whale_txns_cache.get("ts") and (now_ts - _whale_txns_cache["ts"]) < _WHALE_TXNS_TTL:
        cached = dict(_whale_txns_cache.get("response", {}))
        cached["cached"] = True
        return JSONResponse(cached, headers=_CORS)

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
            blocks_r = await _http.get("https://mempool.space/api/v1/blocks/tip/height", timeout=6)
            if blocks_r.status_code != 200:
                return []
            tip = int(blocks_r.text.strip())

            # BTC price from shared cache — avoids extra CoinGecko call
            price_data = await _get_prices(["bitcoin"])
            btc_price = price_data.get("bitcoin", {}).get("usd", 80000) or 80000

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

    # Use shared OKX cache — no duplicate fetch
    txns, futures = await asyncio.gather(fetch_whale_txns(), _get_okx_futures())

    # Netflow signal derived from transaction directions
    sells = sum(1 for t in txns if t["signal"] == "sell")
    buys  = sum(1 for t in txns if t["signal"] == "buy")
    if sells > buys:
        netflow_signal = "sell_pressure"
    elif buys > sells:
        netflow_signal = "accumulation"
    else:
        netflow_signal = "neutral"

    response = {
        "transactions":   txns,
        "netflow_signal": netflow_signal,
        "netflow_sells":  sells,
        "netflow_buys":   buys,
        "futures":        futures,
        "cached":         False,
    }
    # Store in shared cache so /predict/btc-momentum can read netflow without an HTTP call
    _whale_txns_cache["response"] = response
    _whale_txns_cache["netflow"]  = netflow_signal
    _whale_txns_cache["ts"]       = now_ts

    return JSONResponse(response, headers=_CORS)


# ── /predict/polymarket ──────────────────────────────────────────────────────

_polymarket_cache: dict = {}
_POLYMARKET_TTL = 15 * 60  # 15 min


async def handle_polymarket(request: Request) -> JSONResponse:
    """
    GET /predict/polymarket
    Fetches active crypto prediction markets from Polymarket gamma /markets API.
    Filters by crypto keywords, picks markets closest to 50% (most uncertain).
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import time as _time
    import asyncio

    now_ts = _time.time()
    if _polymarket_cache.get("ts") and (now_ts - _polymarket_cache["ts"]) < _POLYMARKET_TTL:
        return JSONResponse(_polymarket_cache["data"], headers=_CORS)

    import re as _re
    # Short tickers use word boundaries to avoid false positives (e.g. "eth" in "Netherlands")
    CRYPTO_KW_PLAIN = [
        "bitcoin", "ethereum", "solana", "crypto", "blockchain", "defi",
        "altcoin", "stablecoin", "microstrategy", "coinbase", "ripple",
        "opensea", "megaeth", "binance", "cardano", "avalanche", "polkadot",
        "chainlink", "uniswap", "arbitrum", "optimism", "polygon", "base chain",
        "memecoin", "nft", "web3", "layer 2", "layer2", "airdrop", "dao",
        "staking", "yield", "dex", "cex", "spot etf", "crypto etf",
    ]
    CRYPTO_KW_WORD = _re.compile(
        r"\b(btc|eth|sol|xrp|bnb|ada|dot|link|matic|avax|uni|arb|op|sei|sui)\b"
    )

    def _is_crypto(q: str) -> str | None:
        """Return matched keyword if q is crypto-related, else None."""
        for kw in CRYPTO_KW_PLAIN:
            if kw in q:
                return kw
        m = CRYPTO_KW_WORD.search(q)
        return m.group() if m else None

    async def fetch_markets(tag: str) -> list:
        url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=100&tag_slug={tag}"
        try:
            log.info("[polymarket] fetching tag=%s url=%s", tag, url)
            r = await _http.get(url, timeout=12)
            log.info("[polymarket] tag=%s status=%s body_len=%s", tag, r.status_code, len(r.content))
            if r.status_code != 200:
                log.warning("[polymarket] non-200 for tag=%s: %s", tag, r.text[:500])
                return []
            data = r.json()
            if not isinstance(data, list):
                log.warning("[polymarket] unexpected response type for tag=%s: %s", tag, str(data)[:200])
                return []
            log.info("[polymarket] tag=%s got %d markets", tag, len(data))
            return data
        except Exception as exc:
            log.exception("[polymarket] fetch failed for tag=%s: %s", tag, exc)
            return []

    def parse_market(m: dict) -> dict | None:
        mid = m.get("id", "?")
        if not isinstance(m, dict):
            log.info("[polymarket] skip %s — not a dict", mid)
            return None
        if m.get("closed"):
            log.info("[polymarket] skip %s — closed", mid)
            return None
        q = (m.get("question") or "").lower()
        kw_match = _is_crypto(q)
        if not kw_match:
            log.info("[polymarket] skip %s — no crypto kw in %r", mid, q[:80])
            return None
        log.info("[polymarket] kw_pass %s — matched %r in %r", mid, kw_match, q[:80])

        raw_prices = m.get("outcomePrices")
        log.info("[polymarket] %s outcomePrices raw=%r", mid, raw_prices)
        try:
            prices = json.loads(raw_prices or "[]")
            if not prices:
                log.info("[polymarket] skip %s — outcomePrices parsed to empty list", mid)
                return None
            yes_prob = round(float(prices[0]) * 100)
            log.info("[polymarket] %s parsed yes_prob=%s", mid, yes_prob)
        except Exception as exc:
            log.info("[polymarket] skip %s — parse error: %s", mid, exc)
            return None
        # Skip effectively resolved (<1% or >99%)
        if yes_prob < 1 or yes_prob > 99:
            log.info("[polymarket] skip %s — yes_prob=%s out of 1-99 range", mid, yes_prob)
            return None
        slug = m.get("slug", "")
        return {
            "question": m.get("question", ""),
            "yes_prob": yes_prob,
            "volume":   round(float(m.get("volume") or 0)),
            "end_date": (m.get("endDate") or "")[:10],
            "url":      f"https://polymarket.com/event/{slug}",
        }

    try:
        pages = await asyncio.gather(
            fetch_markets("crypto"),
            fetch_markets("bitcoin"),
        )
        log.info("[polymarket] pages fetched: %s", [len(p) for p in pages])
        seen: set = set()
        candidates: list = []
        for page in pages:
            for m in page:
                slug = m.get("slug", "")
                if slug in seen:
                    continue
                parsed = parse_market(m)
                if parsed:
                    seen.add(slug)
                    candidates.append(parsed)

        log.info("[polymarket] candidates after filtering: %d", len(candidates))
        # Sort by volume descending
        candidates.sort(key=lambda x: -x["volume"])
        result = {"markets": candidates[:6], "fetched_at": now_ts}
        _polymarket_cache["data"] = result
        _polymarket_cache["ts"]   = now_ts
        return JSONResponse(result, headers=_CORS)

    except Exception as e:
        log.exception("[polymarket] gather failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /predict/manifold ────────────────────────────────────────────────────────

_manifold_cache: dict = {}
_MANIFOLD_TTL = 60 * 60  # 1 hour


async def handle_manifold(request: Request) -> JSONResponse:
    """
    GET /predict/manifold
    Fetches top crypto prediction markets from Manifold Markets (free, no auth).
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import time as _time
    import asyncio

    now_ts = _time.time()
    if _manifold_cache.get("ts") and (now_ts - _manifold_cache["ts"]) < _MANIFOLD_TTL:
        return JSONResponse(_manifold_cache["data"], headers=_CORS)

    TERMS = ["bitcoin", "ethereum", "BTC price", "crypto market", "solana"]

    async def fetch_term(term: str) -> list:
        try:
            log.info("[manifold] fetching term=%r", term)
            r = await _http.get(
                "https://api.manifold.markets/v0/search-markets",
                params={"term": term, "limit": 5, "filter": "open", "sort": "liquidity"},
                timeout=8,
            )
            log.info("[manifold] term=%r status=%s body_len=%s", term, r.status_code, len(r.content))
            if r.status_code != 200:
                log.warning("[manifold] non-200 for term=%r: %s", term, r.text[:500])
                return []
            data = r.json()
            filtered = [
                m for m in data
                if isinstance(m, dict)
                and m.get("outcomeType") == "BINARY"
                and not m.get("isResolved")
                and m.get("probability") is not None
                and m.get("volume", 0) > 500
            ]
            log.info("[manifold] term=%r raw=%d filtered=%d", term, len(data), len(filtered))
            return filtered
        except Exception as exc:
            log.exception("[manifold] fetch failed for term=%r: %s", term, exc)
            return []

    try:
        pages = await asyncio.gather(*[fetch_term(t) for t in TERMS])
        log.info("[manifold] pages fetched: %s", [len(p) for p in pages])
        seen: set = set()
        markets = []
        for page in pages:
            for m in page:
                mid = m.get("id")
                if mid in seen:
                    continue
                seen.add(mid)
                close_ts = m.get("closeTime", 0)
                close_date = ""
                if close_ts and close_ts < 32535212340000:  # skip "never" markets
                    from datetime import datetime, timezone
                    close_date = datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                markets.append({
                    "question":    m.get("question", ""),
                    "probability": round(m.get("probability", 0.5) * 100),
                    "volume":      round(m.get("volume", 0)),
                    "liquidity":   round(m.get("totalLiquidity", 0)),
                    "close_date":  close_date,
                    "url":         m.get("url", ""),
                    "bettors":     m.get("uniqueBettorCount", 0),
                })

        log.info("[manifold] total unique markets: %d", len(markets))
        markets.sort(key=lambda x: -x["liquidity"])
        result = {"markets": markets[:6], "fetched_at": now_ts}
        _manifold_cache["data"] = result
        _manifold_cache["ts"]   = now_ts
        return JSONResponse(result, headers=_CORS)

    except Exception as e:
        log.exception("[manifold] gather failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /predict/metaculus ───────────────────────────────────────────────────────

_metaculus_cache: dict = {}
_METACULUS_TTL = 60 * 60  # 1 hour


async def handle_metaculus(request: Request) -> JSONResponse:
    """
    GET /predict/metaculus
    Fetches open crypto forecasting questions from Metaculus (public API, no auth).
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import time as _time
    import asyncio

    now_ts = _time.time()
    if _metaculus_cache.get("ts") and (now_ts - _metaculus_cache["ts"]) < _METACULUS_TTL:
        return JSONResponse(_metaculus_cache["data"], headers=_CORS)

    SEARCH_TERMS = ["bitcoin", "ethereum", "crypto", "solana", "BTC"]

    async def fetch_term(term: str) -> list:
        try:
            log.info("[metaculus] fetching term=%r", term)
            r = await _http.get(
                "https://www.metaculus.com/api2/questions/",
                params={
                    "search":   term,
                    "status":   "open",
                    "type":     "forecast",
                    "limit":    5,
                    "order_by": "-activity",
                },
                timeout=10,
                headers={"Accept": "application/json"},
            )
            log.info("[metaculus] term=%r status=%s", term, r.status_code)
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("results", [])
        except Exception as exc:
            log.exception("[metaculus] fetch failed for term=%r: %s", term, exc)
            return []

    try:
        pages = await asyncio.gather(*[fetch_term(t) for t in SEARCH_TERMS])
        log.info("[metaculus] pages fetched: %s", [len(p) for p in pages])

        seen: set = set()
        markets = []
        for page in pages:
            for q in page:
                qid = q.get("id")
                if qid in seen:
                    continue
                seen.add(qid)

                # Extract community median probability
                cp = q.get("community_prediction") or {}
                full = cp.get("full") or {}
                q2 = full.get("q2")  # median (0–1)
                if q2 is None:
                    continue
                probability = round(q2 * 100)
                if probability < 1 or probability > 99:
                    continue

                close_time = q.get("close_time", "")
                close_date = ""
                if close_time:
                    try:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                        close_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                markets.append({
                    "question":    q.get("title", ""),
                    "probability": probability,
                    "forecasters": q.get("number_of_forecasters", 0),
                    "close_date":  close_date,
                    "url":         f"https://www.metaculus.com/questions/{qid}/",
                })

        log.info("[metaculus] total unique questions: %d", len(markets))
        markets.sort(key=lambda x: -x["forecasters"])
        result = {"markets": markets[:6], "fetched_at": now_ts}
        _metaculus_cache["data"] = result
        _metaculus_cache["ts"]   = now_ts
        return JSONResponse(result, headers=_CORS)

    except Exception as e:
        log.exception("[metaculus] gather failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── /predict/trending-chips ──────────────────────────────────────────────────

_chips_cache: dict = {}
_CHIPS_TTL = 5 * 60  # 5 min

async def handle_trending_chips(request: Request) -> JSONResponse:
    """
    GET /predict/trending-chips
    Returns 8 hot chip prompts built from live crypto news and trending data:
      1. CryptoCompare latest news headlines  (free, no key)
      2. Reddit r/CryptoCurrency hot posts    (free, no key)
      3. CoinGecko trending coins             (free, no key)
      4. Evergreen fallbacks to pad to 8
    """
    import time
    now_ts = time.time()

    if _chips_cache.get("ts") and (now_ts - _chips_cache["ts"]) < _CHIPS_TTL:
        return JSONResponse(_chips_cache["data"], headers=_CORS)

    chips: list[str] = []
    sources_used: list[str] = []

    def _trim(text: str, limit: int = 52) -> str:
        text = text.strip()
        return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

    # ── 1. CryptoCompare news — top popular headlines ─────────────────────────
    try:
        r = await _http.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=popular",
            timeout=8,
        )
        r.raise_for_status()
        articles = r.json().get("Data", [])[:5]
        for article in articles:
            title = article.get("title", "").strip()
            if title and len(chips) < 4:
                chips.append(_trim(title))
        if chips:
            sources_used.append("cryptocompare")
            log.info("trending-chips: %d chips from CryptoCompare", len(chips))
    except Exception as e:
        log.warning("trending-chips: CryptoCompare failed — %s", e)

    # ── 2. Reddit r/CryptoCurrency hot posts ─────────────────────────────────
    try:
        r = await _http.get(
            "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=10",
            headers={"User-Agent": "EveryCoin/1.0 (crypto research tool)"},
            timeout=8,
        )
        r.raise_for_status()
        posts = r.json().get("data", {}).get("children", [])
        for post in posts:
            data = post.get("data", {})
            # skip pinned/mod posts and very low-engagement posts
            if data.get("stickied") or data.get("score", 0) < 50:
                continue
            title = data.get("title", "").strip()
            if title and len(chips) < 6:
                chips.append(_trim(title))
        if "cryptocompare" not in sources_used or len(chips) > 4:
            sources_used.append("reddit")
            log.info("trending-chips: reddit added, total %d chips", len(chips))
    except Exception as e:
        log.warning("trending-chips: Reddit failed — %s", e)

    # ── 3. CoinGecko trending coins — fill remaining slots ───────────────────
    try:
        r = await _http.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=6,
        )
        r.raise_for_status()
        trending = r.json().get("coins", [])
        templates = [
            "Is {name} a buy right now?",
            "What's driving {name} today?",
            "{symbol} price prediction",
            "Should I hold {name}?",
        ]
        for i, coin in enumerate(trending):
            if len(chips) >= 7:
                break
            item = coin.get("item", {})
            name   = item.get("name", "")
            symbol = item.get("symbol", "")
            if name:
                chips.append(templates[i % len(templates)].format(name=name, symbol=symbol))
        sources_used.append("coingecko")
    except Exception as e:
        log.warning("trending-chips: CoinGecko failed — %s", e)

    # ── 4. Evergreen fallbacks — pad to 8 ────────────────────────────────────
    evergreen = [
        "DeFi market outlook",
        "Best L2s right now",
        "Bear market strategy",
        "How to spot a rug pull",
        "Cold wallet setup guide",
        "Yield farming risks",
        "MEV explained simply",
        "ETH gas tips",
    ]
    for e in evergreen:
        if len(chips) >= 8:
            break
        if e not in chips:
            chips.append(e)
    if not sources_used:
        sources_used.append("evergreen")

    source = "+".join(sources_used)
    result = {"chips": chips[:8], "source": source}
    _chips_cache["data"] = result
    _chips_cache["ts"]   = now_ts
    log.info("trending-chips: returning %d chips (source=%s)", len(chips), source)
    return JSONResponse(result, headers=_CORS)


# ── /predict/market-narrative ────────────────────────────────────────────────

_narrative_cache: dict = {}
_NARRATIVE_TTL = 30 * 60  # 30 minutes


async def handle_market_narrative(request: Request) -> JSONResponse:
    """
    GET /predict/market-narrative
    Returns:
      - narrative: 2-3 sentence AI-written market brief (Claude Haiku)
      - feed:      list of signal items derived from live data
      - direction, blended_score, generated_at, cached
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    import asyncio
    import time as _time
    from datetime import datetime, timezone

    now_ts = _time.time()
    bust   = "t" in request.query_params  # ?t=... busts server cache

    # Serve from cache if still fresh (unless busted)
    if not bust and _narrative_cache.get("ts") and (now_ts - _narrative_cache["ts"]) < _NARRATIVE_TTL:
        data = dict(_narrative_cache["data"])
        data["cached"] = True
        return JSONResponse(data, headers=_CORS)

    try:
        port = os.getenv("PORT", "8000")

        # Gather momentum + whale data in parallel from internal endpoints
        mom_r, whale_r = await asyncio.gather(
            _http.get(f"http://localhost:{port}/predict/btc-momentum", timeout=8),
            _http.get(f"http://localhost:{port}/whale/signals", timeout=8),
            return_exceptions=True,
        )

        mom:   dict = {}
        whale: dict = {}
        if not isinstance(mom_r,   Exception) and mom_r.status_code   == 200: mom   = mom_r.json()
        if not isinstance(whale_r, Exception) and whale_r.status_code == 200: whale = whale_r.json()

        sig          = mom.get("signals", {})
        rsi          = sig.get("rsi",          50)
        macd_hist    = sig.get("macd_hist",     0)
        ret_4h       = sig.get("return_4h",     0)
        ret_24h      = sig.get("return_24h",    0)
        funding_rate = sig.get("funding_rate")
        whale_flow   = sig.get("whale_flow",   "neutral")
        direction    = mom.get("direction",    "HOLD")
        blended      = mom.get("blended_score", 50)
        current_px   = mom.get("current_price", 0)
        target_px    = mom.get("target_price",  0)
        target_pct   = mom.get("target_pct",    0)
        window_h     = mom.get("window_hours",  24)
        confidence   = mom.get("confidence",    0.5)
        ml_score     = mom.get("ml_score",      50)
        macd_val     = sig.get("macd",          0)

        whale_txns = whale.get("transactions", [])
        netflow    = whale.get("netflow_signal", "neutral")
        futures    = whale.get("futures", {})
        oi_usd     = futures.get("open_interest_usd")
        lsr        = futures.get("long_short_ratio")
        oi_trend   = futures.get("oi_trend")

        # ── Build signal feed (Option A) ─────────────────────────────────────
        feed = []
        generated_at = datetime.now(timezone.utc).isoformat()

        # Whale transactions
        for tx in whale_txns[:3]:
            btc  = tx["amount_btc"]
            usd  = tx["amount_usd"]
            frm  = tx["from_label"]
            to   = tx["to_label"]
            sig_ = tx["signal"]
            if sig_ == "sell":
                icon = "🔴"; sentiment = "bearish"
                text = f"{btc:.0f} BTC (${usd/1e6:.1f}M) moved {frm} → {to} — possible sell pressure"
            elif sig_ == "buy":
                icon = "🟢"; sentiment = "bullish"
                text = f"{btc:.0f} BTC (${usd/1e6:.1f}M) moved {frm} → {to} — accumulation signal"
            else:
                icon = "🦈"; sentiment = "neutral"
                text = f"{btc:.0f} BTC (${usd/1e6:.1f}M) moved {frm} → {to}"
            feed.append({"icon": icon, "text": text, "sentiment": sentiment, "time": generated_at})

        # RSI signal
        if rsi < 35:
            feed.append({"icon": "📉", "text": f"RSI at {rsi:.1f} — oversold, potential bounce zone", "sentiment": "bullish", "time": generated_at})
        elif rsi > 65:
            feed.append({"icon": "📈", "text": f"RSI at {rsi:.1f} — overbought, watch for pullback", "sentiment": "bearish", "time": generated_at})

        # MACD histogram
        if macd_hist > 0:
            feed.append({"icon": "⚡", "text": "MACD histogram turning bullish momentum", "sentiment": "bullish", "time": generated_at})
        elif macd_hist < 0:
            feed.append({"icon": "📉", "text": "MACD histogram turning bearish momentum", "sentiment": "bearish", "time": generated_at})

        # 4h return
        if abs(ret_4h) >= 0.5:
            sentiment = "bullish" if ret_4h > 0 else "bearish"
            icon = "📈" if ret_4h > 0 else "📉"
            feed.append({"icon": icon, "text": f"BTC 4h return: {ret_4h:+.2f}% — momentum {'building' if ret_4h > 0 else 'fading'}", "sentiment": sentiment, "time": generated_at})

        # Funding rate
        if funding_rate is not None:
            if funding_rate > 0.05:
                feed.append({"icon": "⚡", "text": f"Funding rate {funding_rate:.4f}% — longs over-leveraged, squeeze risk", "sentiment": "bearish", "time": generated_at})
            elif funding_rate < -0.02:
                feed.append({"icon": "⚡", "text": f"Funding rate {funding_rate:.4f}% — shorts over-leveraged, potential squeeze up", "sentiment": "bullish", "time": generated_at})

        # Open interest trend
        if oi_trend == "down":
            feed.append({"icon": "📊", "text": "Open interest falling — positions closing, trend may be weakening", "sentiment": "bearish", "time": generated_at})
        elif oi_trend == "up":
            feed.append({"icon": "📊", "text": "Open interest rising — new money entering the market", "sentiment": "bullish", "time": generated_at})

        # Long/short ratio
        if lsr is not None:
            if lsr < 0.8:
                feed.append({"icon": "🔄", "text": f"L/S ratio {lsr:.3f} — more shorts than longs, squeeze setup possible", "sentiment": "bullish", "time": generated_at})
            elif lsr > 1.3:
                feed.append({"icon": "🔄", "text": f"L/S ratio {lsr:.3f} — longs dominating, elevated liquidation risk", "sentiment": "bearish", "time": generated_at})

        # Netflow
        if netflow == "sell_pressure":
            feed.append({"icon": "🏦", "text": "Whale netflow: BTC moving to exchanges — distribution signal", "sentiment": "bearish", "time": generated_at})
        elif netflow == "accumulation":
            feed.append({"icon": "🏦", "text": "Whale netflow: BTC leaving exchanges — accumulation signal", "sentiment": "bullish", "time": generated_at})

        feed = feed[:6]  # cap at 6 items

        # ── Build narrative prompt ────────────────────────────────────────────
        price_str    = f"${current_px:,.0f}" if current_px else "unknown"
        target_str   = f"${target_px:,.0f} ({target_pct:+.2f}%)" if target_px else "uncertain"
        lsr_str      = f"{lsr:.3f}" if lsr is not None else "N/A"
        funding_str  = f"{funding_rate:.4f}%" if funding_rate is not None else "N/A"

        prompt = f"""You are a concise crypto market analyst. Write exactly 2-3 sentences summarising current BTC market conditions for a retail trader. Be specific with numbers. Do not use markdown, headers, or bullet points.

Data:
- BTC price: {price_str}
- AI blended score: {blended}/100, direction: {direction}
- ML model score: {ml_score}/100, confidence: {int(confidence*100)}%
- Target: {target_str} over {window_h}h
- RSI-14: {rsi:.1f}, MACD: {macd_val:.2f}, MACD hist: {macd_hist:.2f}
- 4h return: {ret_4h:+.2f}%, 24h return: {ret_24h:+.2f}%
- Whale flow: {whale_flow}, netflow: {netflow}
- OKX funding rate: {funding_str}
- Long/short ratio: {lsr_str}
- Open interest trend: {oi_trend or 'unknown'}

Write your 2-3 sentence market brief now:"""

        # ── Call Claude Haiku ─────────────────────────────────────────────────
        narrative = ""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256, temperature=0.4)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            narrative = response.content.strip()
        except Exception:
            log.exception("Claude narrative generation failed")
            # Fallback: construct narrative from data
            dir_word = "bullish" if direction == "BUY" else ("bearish" if direction == "SELL" else "neutral")
            narrative = (
                f"BTC is trading at {price_str} with a {dir_word} AI signal (score {blended}/100). "
                f"RSI at {rsi:.1f} and {ret_24h:+.2f}% 24h return with {netflow.replace('_', ' ')} whale activity. "
                f"Short-term target {target_str} over {window_h}h."
            )

        result = {
            "narrative":     narrative,
            "feed":          feed,
            "direction":     direction,
            "blended_score": blended,
            "generated_at":  generated_at,
            "cached":        False,
        }

        _narrative_cache["data"] = result
        _narrative_cache["ts"]   = now_ts

        return JSONResponse(result, headers=_CORS)

    except Exception as e:
        log.exception("market-narrative failed")
        return JSONResponse({"error": str(e)}, status_code=500, headers=_CORS)


# ── Blog comments — SQLite storage ───────────────────────────────────────────

import sqlite3
import re
import time as _time_mod

_COMMENTS_DB = os.path.join(os.path.dirname(__file__), "data", "comments.db")


def _comments_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_COMMENTS_DB), exist_ok=True)
    conn = sqlite3.connect(_COMMENTS_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            slug      TEXT NOT NULL,
            name      TEXT NOT NULL,
            text      TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_slug ON comments(slug)")
    conn.commit()
    return conn


_comments_db: sqlite3.Connection | None = None


def _get_comments_db() -> sqlite3.Connection:
    global _comments_db
    if _comments_db is None:
        _comments_db = _comments_conn()
    return _comments_db


async def handle_blog_comments(request: Request) -> JSONResponse:
    """
    GET  /blog/comments?slug=<post-slug>  — list approved comments
    POST /blog/comments                   — submit a new comment
         body: { slug, name, text }
    OPTIONS /blog/comments                — CORS preflight
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_CORS)

    db = _get_comments_db()

    if request.method == "GET":
        slug = request.query_params.get("slug", "").strip()
        if not slug:
            return JSONResponse({"error": "slug param required"}, status_code=400, headers=_CORS)
        rows = db.execute(
            "SELECT id, name, text, created_at FROM comments WHERE slug = ? ORDER BY created_at ASC",
            (slug,),
        ).fetchall()
        comments = [{"id": r["id"], "name": r["name"], "text": r["text"], "created_at": r["created_at"]} for r in rows]
        return JSONResponse({"comments": comments}, headers=_CORS)

    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400, headers=_CORS)

        slug = (body.get("slug") or "").strip()
        name = (body.get("name") or "").strip()[:80]
        text = (body.get("text") or "").strip()[:2000]

        if not slug or not name or not text:
            return JSONResponse({"error": "slug, name, and text are required"}, status_code=400, headers=_CORS)
        if not re.match(r'^[a-z0-9\-]+$', slug):
            return JSONResponse({"error": "Invalid slug"}, status_code=400, headers=_CORS)

        created_at = int(_time_mod.time())
        cur = db.execute(
            "INSERT INTO comments (slug, name, text, created_at) VALUES (?, ?, ?, ?)",
            (slug, name, text, created_at),
        )
        db.commit()
        return JSONResponse({"id": cur.lastrowid, "name": name, "text": text, "created_at": created_at}, headers=_CORS)

    return JSONResponse({"error": "Method not allowed"}, status_code=405, headers=_CORS)


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
        Route("/predict/btc-momentum", handle_btc_momentum, methods=["GET", "OPTIONS"]),
        Route("/predict/price-target", handle_price_target, methods=["GET", "OPTIONS"]),
        Route("/predict/btc-journey", handle_btc_journey, methods=["GET", "OPTIONS"]),
        Route("/predict/price-history", handle_price_history, methods=["GET", "OPTIONS"]),
        Route("/whale/signals", handle_whale_signals, methods=["GET", "OPTIONS"]),
        Route("/predict/polymarket",        handle_polymarket,        methods=["GET", "OPTIONS"]),
        Route("/predict/manifold",          handle_manifold,          methods=["GET", "OPTIONS"]),
        Route("/predict/metaculus",         handle_metaculus,         methods=["GET", "OPTIONS"]),
        Route("/predict/trending-chips",    handle_trending_chips,    methods=["GET", "OPTIONS"]),
        Route("/predict/market-narrative", handle_market_narrative, methods=["GET", "OPTIONS"]),
        Route("/blog/comments", handle_blog_comments, methods=["GET", "POST", "OPTIONS"]),
        Route("/health", handle_health, methods=["GET"]),
    ],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info("EveryCoin server starting on http://0.0.0.0:%d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
