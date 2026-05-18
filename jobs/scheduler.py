"""
jobs/scheduler.py
-----------------
APScheduler wiring for PatGPT background jobs.

Schedule:
  Weekly  — every Monday at 08:00 UTC  → weekly_report.run_weekly_report()
  Monthly — 1st of each month 02:00 UTC → monthly_retrain.run_monthly_retrain()

Usage A — embedded in FastAPI (recommended):
    Add to api/server.py startup:

        from jobs.scheduler import start_scheduler, stop_scheduler

        # At startup (after all engines are loaded):
        start_scheduler()

        # At shutdown:
        stop_scheduler()

Usage B — standalone cron wrapper (no FastAPI dependency):
    python jobs/scheduler.py
    # Runs in foreground, blocking. Use with nohup or a process manager.

Usage C — run a specific job immediately:
    python jobs/scheduler.py --job weekly
    python jobs/scheduler.py --job monthly
    python jobs/scheduler.py --job monthly --no-reload
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("patgpt.scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ── APScheduler ───────────────────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    _APScheduler_AVAILABLE = True
except ImportError:
    _APScheduler_AVAILABLE = False
    # Non-fatal — server still starts, jobs just run manually or via cron.
    logger.warning(
        "apscheduler not installed — background scheduler disabled."
        "  Fix: pip install apscheduler"
        "  Jobs can still be run manually:"
        "    python jobs/scheduler.py --job weekly"
        "    python jobs/scheduler.py --job monthly --no-reload"
    )

# ── Job imports ───────────────────────────────────────────────────────────────
from jobs.weekly_report  import run_weekly_report
from jobs.monthly_retrain import run_monthly_retrain

# ── Module-level scheduler singleton ─────────────────────────────────────────
_scheduler: "BackgroundScheduler | None" = None


def _safe_weekly() -> None:
    """Wrapper with exception guard so one failure doesn't kill the scheduler."""
    try:
        logger.info("▶  Running weekly prompt performance report…")
        report = run_weekly_report()
        flagged = report.get("summary", {}).get("flagged_versions", [])
        if flagged:
            logger.warning(f"⚠  Prompt versions flagged for review: {flagged}")
        else:
            logger.info("✅ Weekly report complete — no versions flagged.")
    except Exception as e:
        logger.error(f"Weekly report failed: {e}", exc_info=True)


def _safe_monthly() -> None:
    """Wrapper with exception guard."""
    try:
        logger.info("▶  Running monthly ML retraining…")
        result = run_monthly_retrain(skip_server_reload=False)
        if result.get("overall_success"):
            logger.info("✅ Monthly retraining complete.")
        else:
            logger.error("❌ Monthly retraining reported failure — check retrain_log.jsonl")
    except Exception as e:
        logger.error(f"Monthly retraining failed: {e}", exc_info=True)


def start_scheduler() -> None:
    """
    Start the APScheduler BackgroundScheduler.
    Call this once at server startup.
    """
    global _scheduler

    if not _APScheduler_AVAILABLE:
        logger.warning("APScheduler not available — background jobs disabled.")
        return

    if _scheduler and _scheduler.running:
        logger.info("Scheduler already running — skipping.")
        return

    _scheduler = BackgroundScheduler(timezone="UTC")

    # ── Weekly: every Monday at 08:00 UTC ────────────────────────────────
    _scheduler.add_job(
        _safe_weekly,
        trigger=CronTrigger(day_of_week="mon", hour=8, minute=0),
        id="weekly_report",
        name="Weekly Prompt Performance Report",
        replace_existing=True,
        misfire_grace_time=3600,   # allow up to 1h late if server was down
    )

    # ── Monthly: 1st of each month at 02:00 UTC ──────────────────────────
    _scheduler.add_job(
        _safe_monthly,
        trigger=CronTrigger(day=1, hour=2, minute=0),
        id="monthly_retrain",
        name="Monthly ML Retraining",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    logger.info("✅ Scheduler started.")
    logger.info("   Jobs registered:")
    for job in _scheduler.get_jobs():
        logger.info(f"     {job.id:20s} — next run: {job.next_run_time}")


def stop_scheduler() -> None:
    """
    Gracefully shut down the scheduler.
    Call this at server shutdown.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


def trigger_now(job_id: str) -> None:
    """Manually trigger a job immediately (useful for testing)."""
    if not _scheduler or not _scheduler.running:
        logger.warning("Scheduler not running — starting temporarily.")
        start_scheduler()
    _scheduler.get_job(job_id).modify(next_run_time=__import__("datetime").datetime.utcnow())
    logger.info(f"Job '{job_id}' triggered immediately.")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MODE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatGPT Background Job Scheduler")
    parser.add_argument(
        "--job", choices=["weekly", "monthly", "all"],
        help="Run a specific job immediately and exit, instead of starting the scheduler loop."
    )
    parser.add_argument(
        "--no-reload", action="store_true",
        help="(monthly only) Skip POST /admin/reload_models after training."
    )
    args = parser.parse_args()

    if args.job == "weekly":
        logger.info("Running weekly report immediately…")
        _safe_weekly()
        sys.exit(0)

    elif args.job == "monthly":
        logger.info("Running monthly retraining immediately…")
        try:
            result = run_monthly_retrain(skip_server_reload=args.no_reload)
            sys.exit(0 if result.get("overall_success") else 1)
        except Exception as e:
            logger.error(f"Monthly retrain failed: {e}", exc_info=True)
            sys.exit(1)

    else:
        # Default: start scheduler loop and block
        if not _APScheduler_AVAILABLE:
            logger.error("Cannot start scheduler — apscheduler not installed.")
            logger.error("Run:  pip install apscheduler")
            sys.exit(1)

        start_scheduler()
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            stop_scheduler()
            logger.info("Scheduler stopped.")