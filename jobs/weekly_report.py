"""
jobs/weekly_report.py
---------------------
Weekly background job — Prompt Performance Report.

Reads:
  data/feedback_log.jsonl   — outcome feedback per interaction
  logs/llm_calls.jsonl      — every LLM call with prompt_version + doctor_id

Writes:
  reports/prompt_performance.json — per-version win rate, flagged versions

Win rate definition:
  win_rate = (feedback records with outcome == "positive") /
             (all feedback records for that prompt_version)

Auto-flags any prompt version with win_rate < 0.30 for human review.

Also computes:
  - avg_interest_lift_after_visit  (interest_after - baseline avg_interest)
  - aida_stage_progression_rate    (% of interactions where aida advanced)

Run standalone:
    python jobs/weekly_report.py

Or called by jobs/scheduler.py every Monday at 08:00.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths (mirror server.py constants) ───────────────────────────────────────
DATA_DIR     = Path(os.getenv("DATA_DIR",    "data/"))
LOGS_DIR     = Path(os.getenv("LOGS_DIR",    "logs/"))
REPORTS_DIR  = Path(os.getenv("REPORTS_DIR", "reports/"))
FEEDBACK_LOG = DATA_DIR  / "feedback_log.jsonl"
LLM_LOG      = LOGS_DIR  / "llm_calls.jsonl"
REPORT_OUT   = REPORTS_DIR / "prompt_performance.json"

WIN_RATE_FLAG_THRESHOLD = 0.30   # flag versions below this


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _compute_win_rates(feedback: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Group feedback by prompt_version and compute win/loss counts + win_rate.
    """
    by_version: Dict[str, List[Dict]] = defaultdict(list)
    for rec in feedback:
        pv = rec.get("prompt_version", "unknown")
        by_version[pv].append(rec)

    result = {}
    for pv, records in by_version.items():
        total    = len(records)
        wins     = sum(1 for r in records if str(r.get("outcome", "")).lower() == "positive")
        losses   = sum(1 for r in records if str(r.get("outcome", "")).lower() == "negative")
        neutral  = total - wins - losses
        win_rate = round(wins / total, 4) if total > 0 else 0.0

        # Interest lift: interest_after minus a baseline of 2.5 (mid-scale)
        # A positive lift means the visit raised doctor interest.
        interest_values = [
            float(r["interest_after"]) for r in records
            if r.get("interest_after") is not None
        ]
        avg_interest_after = round(sum(interest_values) / len(interest_values), 3) \
            if interest_values else None
        avg_interest_lift  = round(avg_interest_after - 2.5, 3) \
            if avg_interest_after is not None else None

        flagged = win_rate < WIN_RATE_FLAG_THRESHOLD

        result[pv] = {
            "prompt_version":          pv,
            "total_interactions":      total,
            "wins":                    wins,
            "losses":                  losses,
            "neutral":                 neutral,
            "win_rate":                win_rate,
            "avg_interest_after":      avg_interest_after,
            "avg_interest_lift":       avg_interest_lift,
            "flagged_for_review":      flagged,
            "flag_reason":             f"win_rate {win_rate:.0%} < {WIN_RATE_FLAG_THRESHOLD:.0%} threshold"
                                       if flagged else None,
        }
    return result


def _compute_call_stats(llm_calls: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate LLM call telemetry from llm_calls.jsonl.
    """
    if not llm_calls:
        return {}

    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for rec in llm_calls:
        by_method[rec.get("method", "unknown")].append(rec)

    stats = {}
    for method, calls in by_method.items():
        latencies    = [c["latency_ms"] for c in calls if c.get("latency_ms")]
        token_counts = [c["token_count"] for c in calls if c.get("token_count")]
        success_rate = sum(1 for c in calls if c.get("success")) / len(calls) if calls else 0

        stats[method] = {
            "total_calls":        len(calls),
            "success_rate":       round(success_rate, 4),
            "avg_latency_ms":     round(sum(latencies) / len(latencies), 1) if latencies else None,
            "p95_latency_ms":     round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
                                  if len(latencies) >= 20 else None,
            "avg_tokens":         round(sum(token_counts) / len(token_counts), 1)
                                  if token_counts else None,
        }
    return stats


def _compute_aida_progression(feedback: List[Dict]) -> float:
    """
    Estimate AIDA stage progression rate.
    A visit "progressed" if interest_after >= 4 (desire/action signal)
    and outcome was positive — rough proxy without storing pre/post AIDA.
    Returns fraction of interactions showing progression signal.
    """
    if not feedback:
        return 0.0
    progression = sum(
        1 for r in feedback
        if str(r.get("outcome", "")).lower() == "positive"
        and (r.get("interest_after") or 0) >= 4
    )
    return round(progression / len(feedback), 4)


def run_weekly_report() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feedback  = _read_jsonl(FEEDBACK_LOG)
    llm_calls = _read_jsonl(LLM_LOG)

    print(f"[weekly_report] Feedback records:  {len(feedback)}")
    print(f"[weekly_report] LLM call records:  {len(llm_calls)}")

    # ── Last-7-days slice ─────────────────────────────────────────────────
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent_feedback = [
        r for r in feedback
        if _parse_dt(r.get("logged_at")) >= cutoff
    ]
    print(f"[weekly_report] Feedback in last 7 days: {len(recent_feedback)}")

    win_rates  = _compute_win_rates(feedback)          # all-time
    recent_wr  = _compute_win_rates(recent_feedback)   # last 7 days
    call_stats = _compute_call_stats(llm_calls)
    aida_prog  = _compute_aida_progression(feedback)

    flagged_versions = [
        pv for pv, data in win_rates.items()
        if data["flagged_for_review"]
    ]

    report = {
        "generated_at":              datetime.utcnow().isoformat(),
        "period":                    "all-time + last_7_days",
        "summary": {
            "total_feedback_records":    len(feedback),
            "recent_feedback_7d":        len(recent_feedback),
            "prompt_versions_tracked":   len(win_rates),
            "flagged_versions":          flagged_versions,
            "overall_win_rate":          round(
                sum(d["wins"] for d in win_rates.values()) /
                max(sum(d["total_interactions"] for d in win_rates.values()), 1), 4
            ),
            "aida_stage_progression_rate": aida_prog,
        },
        "per_version_all_time":      win_rates,
        "per_version_last_7d":       recent_wr,
        "llm_call_stats":            call_stats,
    }

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[weekly_report] Report written → {REPORT_OUT}")

    if flagged_versions:
        print(f"[weekly_report] ⚠  FLAGGED for human review: {flagged_versions}")
    else:
        print("[weekly_report] ✅ All prompt versions above win-rate threshold.")

    return report


def _parse_dt(s: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return datetime.min


if __name__ == "__main__":
    run_weekly_report()