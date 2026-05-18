"""
jobs/monthly_retrain.py
-----------------------
Monthly background job — ML Model Retraining.

Steps:
  1. Run ml_training.py as a subprocess (or directly via importlib)
     against the latest data CSV.
  2. POST /admin/reload_models to hot-reload models in the running server
     without a restart.
  3. Write a retraining log entry to logs/retrain_log.jsonl.

Can also run headlessly (no live server) — in that case step 2 is skipped
and a reminder is printed.

Run standalone:
    python jobs/monthly_retrain.py

Or called by jobs/scheduler.py on the 1st of each month at 02:00.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.getenv("DATA_DIR",    "data/"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  "models/"))
LOGS_DIR    = Path(os.getenv("LOGS_DIR",    "logs/"))
CSV_PATH    = DATA_DIR / "doctor_sales_dummy_data.csv"
RETRAIN_LOG = LOGS_DIR / "retrain_log.jsonl"

# Server base URL for hot-reload call (set via env or default to localhost)
SERVER_URL  = os.getenv("PATGPT_SERVER_URL", "http://localhost:8000")
TRAINING_SCRIPT = ROOT / "main" / "ml" / "ml_training.py"


def _log_retrain(entry: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_training() -> dict:
    """
    Run ml_training.py as a subprocess.
    Returns a dict with success flag, stdout, stderr, and elapsed seconds.
    """
    print(f"[monthly_retrain] Starting training: {TRAINING_SCRIPT}")
    print(f"[monthly_retrain] CSV:    {CSV_PATH}")
    print(f"[monthly_retrain] Models: {MODELS_DIR}")

    # Force UTF-8 stdout/stderr so Windows cp1252 does not crash on emoji.
    import copy
    env = copy.copy(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--csv",        str(CSV_PATH),
            "--models-dir", str(MODELS_DIR),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    elapsed = round(time.time() - t0, 1)

    success = result.returncode == 0
    if success:
        print(f"[monthly_retrain] ✅ Training completed in {elapsed}s")
    else:
        print(f"[monthly_retrain] ❌ Training FAILED (exit code {result.returncode})")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")

    return {
        "success":        success,
        "elapsed_sec":    elapsed,
        "returncode":     result.returncode,
        "stdout_tail":    result.stdout[-1000:] if result.stdout else "",
        "stderr_tail":    result.stderr[-1000:] if result.stderr else "",
    }


def hot_reload_server() -> dict:
    """
    POST /admin/reload_models to the running server.
    Returns response dict or error dict.
    """
    try:
        import httpx  # type: ignore
        print(f"[monthly_retrain] Hot-reloading models via {SERVER_URL}/admin/reload_models …")
        resp = httpx.post(f"{SERVER_URL}/admin/reload_models", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[monthly_retrain] ✅ Hot-reload successful: {data.get('status')}")
            return {"success": True, "response": data}
        else:
            print(f"[monthly_retrain] ⚠  Hot-reload returned {resp.status_code}: {resp.text[:200]}")
            return {"success": False, "status_code": resp.status_code, "body": resp.text[:200]}
    except ImportError:
        print("[monthly_retrain] httpx not installed — skipping hot-reload call.")
        print("                  Run: pip install httpx")
        print("                  Or manually call POST /admin/reload_models after training.")
        return {"success": False, "error": "httpx not installed"}
    except Exception as e:
        print(f"[monthly_retrain] ⚠  Hot-reload call failed: {e}")
        print("                  Server may not be running — models will load on next restart.")
        return {"success": False, "error": str(e)}


def run_monthly_retrain(skip_server_reload: bool = False) -> dict:
    started_at = datetime.utcnow().isoformat()
    print(f"\n{'='*60}")
    print(f"[monthly_retrain] Starting — {started_at}")
    print(f"{'='*60}\n")

    # ── Step 1: train ─────────────────────────────────────────────────────
    train_result = run_training()

    # ── Step 2: hot-reload ────────────────────────────────────────────────
    reload_result = {}
    if train_result["success"] and not skip_server_reload:
        reload_result = hot_reload_server()
    elif not train_result["success"]:
        print("[monthly_retrain] Skipping hot-reload — training failed.")
    elif skip_server_reload:
        print("[monthly_retrain] Skipping hot-reload (skip_server_reload=True).")

    # ── Step 3: log ───────────────────────────────────────────────────────
    log_entry = {
        "started_at":     started_at,
        "completed_at":   datetime.utcnow().isoformat(),
        "training":       train_result,
        "hot_reload":     reload_result,
        "overall_success": train_result["success"],
    }
    _log_retrain(log_entry)
    print(f"\n[monthly_retrain] Log → {RETRAIN_LOG}")

    # Read and surface the training manifest metrics
    manifest_path = MODELS_DIR / "training_manifest.json"
    if manifest_path.exists() and train_result["success"]:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            models_meta = manifest.get("models", {})
            print("\n[monthly_retrain] Model metrics from this run:")
            for name, meta in models_meta.items():
                if not meta:
                    continue
                auc  = meta.get("auc_roc")
                f1   = meta.get("macro_f1")
                sil  = meta.get("silhouette")
                metric_str = (
                    f"AUC={auc}" if auc else
                    f"F1={f1}"   if f1  else
                    f"Sil={sil}" if sil else "no metric"
                )
                print(f"  {name:20s}: {metric_str}")
        except Exception as e:
            print(f"[monthly_retrain] Could not read manifest: {e}")

    return log_entry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PatGPT Monthly Retraining")
    parser.add_argument(
        "--no-reload", action="store_true",
        help="Skip the POST /admin/reload_models call (use when server is not running)"
    )
    args = parser.parse_args()
    run_monthly_retrain(skip_server_reload=args.no_reload)