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

from mlops import db
from mlops.config import FETCH_INTERVAL_MINUTES, BACKFILL_FLAG
from mlops.features import run_feature_engineering, run_feature_engineering_full
from mlops.fetch import run_pipeline
from mlops.backfill import run_backfill
from mlops.train import train as run_train

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
    log.info("  Hourly fetch  : every %d min", args.interval)
    log.info("  Weekly retrain: every Sunday 02:00 UTC")

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
