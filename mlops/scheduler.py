"""
MLOps pipeline scheduler.

Usage:
  # Run once immediately (useful for testing):
  python -m mlops.scheduler --once

  # Run continuously on a schedule (default: every 60 min):
  python -m mlops.scheduler

  # Custom interval:
  python -m mlops.scheduler --interval 30
"""

import argparse
import asyncio
import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from mlops import db
from mlops.config import FETCH_INTERVAL_MINUTES
from mlops.fetch import run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("everycoin.mlops.scheduler")


def _tick() -> None:
    """Synchronous wrapper — APScheduler calls this; it runs the async pipeline."""
    asyncio.run(run_pipeline())


def main() -> None:
    parser = argparse.ArgumentParser(description="EveryCoin MLOps data pipeline")
    parser.add_argument("--once", action="store_true", help="Run pipeline once then exit")
    parser.add_argument("--interval", type=int, default=FETCH_INTERVAL_MINUTES, help="Fetch interval in minutes")
    args = parser.parse_args()

    db.init_db()

    if args.once:
        log.info("Running pipeline once ...")
        _tick()
        counts = db.row_counts()
        log.info("Done. DB row counts: %s", counts)
        return

    log.info("Starting scheduler — interval: %d min", args.interval)
    _tick()  # run immediately on startup

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(_tick, "interval", minutes=args.interval, id="pipeline")
    log.info("Next run in %d minutes. Press Ctrl+C to stop.", args.interval)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
