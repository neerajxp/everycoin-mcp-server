"""
MLOps pipeline scheduler.

Jobs:
  Hourly  — fetch market data + compute features
  Weekly  — retrain XGBoost model (every Sunday 02:00 UTC)

Usage:
  python -m mlops.scheduler              # run both jobs on schedule
  python -m mlops.scheduler --once       # run fetch+features once then exit
  python -m mlops.scheduler --retrain    # run retrain once then exit
  python -m mlops.scheduler --interval 30  # custom fetch interval (minutes)
"""

import argparse
import asyncio
import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

import httpx

from mlops import db
from mlops.config import FETCH_INTERVAL_MINUTES, BACKFILL_FLAG
from mlops.features import run_feature_engineering, run_feature_engineering_full
from mlops.fetch import run_pipeline
from mlops.backfill import run_backfill
from mlops.train import train as run_train
from mlops.serve import predict as ml_predict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("everycoin.mlops.scheduler")


def _backfill_once() -> None:
    """Run 90-day backfill + full feature engineering exactly once.
    Uses flag file (persistent volume) or row count as fallback guard."""
    if BACKFILL_FLAG.exists():
        log.info("Backfill already done (flag found) — skipping")
        return
    counts = db.row_counts()
    if counts["price_history"] > 100:
        log.info("DB already populated (%d rows) — skipping backfill", counts["price_history"])
        BACKFILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        BACKFILL_FLAG.touch()
        return
    log.info("=== First boot: running 90-day backfill ===")
    try:
        asyncio.run(run_backfill(days=90))
        log.info("Backfill complete — computing features for all historical rows ...")
        run_feature_engineering_full()
        BACKFILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        BACKFILL_FLAG.touch()
        log.info("=== Backfill done | %s ===", db.row_counts())
    except Exception as e:
        log.error("=== Backfill FAILED: %s — will retry on next restart ===", e)


def _tick() -> None:
    """Hourly: fetch data + compute features."""
    asyncio.run(run_pipeline())
    run_feature_engineering()


def _daily_prediction() -> None:
    """
    Daily at 9pm EST (02:00 UTC next day):
    1. Settle yesterday's prediction (record actual price + Hit/Miss)
    2. Make today's 24h price prediction for BTC and store it
    """
    log.info("=== Daily price prediction job triggered ===")
    coin_id = "bitcoin"

    try:
        # ── Step 1: settle yesterday's prediction ──────────────────────────────
        prev = db.latest_prediction(coin_id)
        if prev and not prev.get("actual_price"):
            # Fetch current BTC price to use as "actual" for yesterday's target
            resp = httpx.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10,
            )
            if resp.status_code == 200:
                actual = resp.json().get(coin_id, {}).get("usd")
                if actual:
                    db.update_prediction_outcome(prev["id"], actual)
                    log.info("Settled prediction id=%d target=$%.2f actual=$%.2f",
                             prev["id"], prev["predicted_price"], actual)

        # ── Step 2: make today's prediction ────────────────────────────────────
        pred = ml_predict(coin_id)
        if pred.get("error") or not pred.get("signal"):
            log.warning("Skipping prediction — no signal: %s", pred.get("error"))
            return

        signal     = pred["signal"]
        ai_score   = pred["ai_score"]
        confidence = pred["confidence"]
        return_1h  = signal.get("return_1h", 0)

        raw_move_pct     = return_1h * 8
        conviction       = (ai_score - 50) / 50
        blended_pct      = (raw_move_pct * 0.6) + (conviction * abs(raw_move_pct) * 0.4 + conviction * 2)
        predicted_move   = max(-15.0, min(15.0, round(blended_pct * confidence, 2)))
        predicted_score  = min(100, max(0, round(ai_score + conviction * 10 * confidence)))

        # Fetch current price
        resp = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Price fetch failed — skipping prediction")
            return

        current_price   = resp.json().get(coin_id, {}).get("usd")
        predicted_price = round(current_price * (1 + predicted_move / 100), 2)

        db.insert_price_prediction(
            coin_id=coin_id,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_move_pct=predicted_move,
            predicted_score=predicted_score,
            confidence=confidence,
        )
        log.info("=== Daily prediction saved: target=$%.2f move=%.2f%% ===",
                 predicted_price, predicted_move)

    except Exception as e:
        log.error("=== Daily prediction FAILED: %s ===", e)


def _weekly_retrain() -> None:
    """Weekly: retrain XGBoost on latest feature_store data and log to MLflow."""
    log.info("=== Weekly retrain triggered ===")
    try:
        metrics = run_train()
        log.info("=== Weekly retrain complete | roc_auc=%.4f accuracy=%.4f ===",
                 metrics.get("roc_auc", 0), metrics.get("accuracy", 0))
    except Exception as e:
        log.error("=== Weekly retrain FAILED: %s ===", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="EveryCoin MLOps scheduler")
    parser.add_argument("--once",     action="store_true", help="Run fetch+features once then exit")
    parser.add_argument("--retrain",  action="store_true", help="Run retrain once then exit")
    parser.add_argument("--interval", type=int, default=FETCH_INTERVAL_MINUTES, help="Fetch interval in minutes")
    args = parser.parse_args()

    db.init_db()

    # ── One-shot modes ────────────────────────────────────────────────────────
    if args.once:
        log.info("Running fetch + features once ...")
        _tick()
        log.info("Done. DB: %s", db.row_counts())
        return

    if args.retrain:
        log.info("Running retrain once ...")
        _weekly_retrain()
        return

    # ── Continuous scheduled mode ─────────────────────────────────────────────
    log.info("Starting scheduler")
    log.info("  Hourly fetch      : every %d min", args.interval)
    log.info("  Daily prediction  : every day 02:00 UTC (9pm EST)")
    log.info("  Weekly retrain    : every Sunday 02:00 UTC")

    _backfill_once()  # no-op if flag exists, runs once on fresh volume
    _tick()           # run first fetch immediately

    scheduler = BlockingScheduler(timezone="UTC")

    # Hourly fetch + features
    scheduler.add_job(
        _tick,
        "interval",
        minutes=args.interval,
        id="hourly_pipeline",
    )

    # Daily prediction — every day at 02:00 UTC (= 9pm EST / 10pm EDT)
    scheduler.add_job(
        _daily_prediction,
        "cron",
        hour=2,
        minute=0,
        id="daily_prediction",
    )

    # Weekly retrain — every Sunday at 02:00 UTC
    scheduler.add_job(
        _weekly_retrain,
        "cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        id="weekly_retrain",
    )

    log.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
