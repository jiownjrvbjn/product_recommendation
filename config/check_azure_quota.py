"""
check_azure_quota.py
--------------------
Run this ONCE before embedding to discover your actual Azure TPM limit
and get the optimal MAX_BATCH_SIZE — no Azure Portal needed.

Usage:
    python check_azure_quota.py

Reads from your existing .env file (same as the main project).
"""

import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ── Connect using your existing env vars ──────────────────────────────────────
client = AzureOpenAI(
    api_key        = os.getenv("EMBEDDING_API_KEY"),
    azure_endpoint = os.getenv("EMBEDDING_ENDPOINT"),
    api_version    = os.getenv("EMBEDDING_API_VERSION", "2024-02-01"),
)
EMBED_MODEL = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# ── Send a tiny probe request and read the response headers ───────────────────
PROBE_TEXT = ["quota check probe"]   # 4 tokens — negligible cost

print(f"\n🔍 Probing Azure deployment: {EMBED_MODEL}")
print("   Sending 1-token test request to read quota headers...\n")

raw_response = client.embeddings.with_raw_response.create(
    model = EMBED_MODEL,
    input = PROBE_TEXT,
)

headers = dict(raw_response.headers)

# ── Extract the quota headers Azure sends back ────────────────────────────────
# Header reference:
#   x-ratelimit-limit-tokens        → your TPM ceiling
#   x-ratelimit-remaining-tokens    → tokens left in current window
#   x-ratelimit-limit-requests      → your RPM ceiling
#   x-ratelimit-remaining-requests  → requests left in current window
#   x-ms-region                     → which Azure region served the request

tpm_limit       = int(headers.get("x-ratelimit-limit-tokens",      0))
tpm_remaining   = int(headers.get("x-ratelimit-remaining-tokens",   0))
rpm_limit       = int(headers.get("x-ratelimit-limit-requests",     0))
rpm_remaining   = int(headers.get("x-ratelimit-remaining-requests", 0))
region          = headers.get("x-ms-region", "unknown")

tpm_used_pct    = ((tpm_limit - tpm_remaining) / tpm_limit * 100) if tpm_limit else 0
rpm_used_pct    = ((rpm_limit - rpm_remaining) / rpm_limit * 100) if rpm_limit else 0

# ── Calculate optimal batch size from real TPM ────────────────────────────────
AVG_TOKENS_PER_CHUNK = 300    # 1200-char chunks ÷ 4 chars/token
SAFETY_MARGIN        = 0.80   # stay at 80% of quota to absorb spikes
SLEEP_SECONDS        = 62     # slightly over 60s for TPM window to fully reset

safe_tokens_per_req  = int(tpm_limit * SAFETY_MARGIN)
optimal_batch_size   = max(1, safe_tokens_per_req // AVG_TOKENS_PER_CHUNK)
optimal_batch_size   = min(optimal_batch_size, 2047)   # hard API ceiling

# Time estimates for your coverage_data1 (189,093 chunks)
CHUNKS_TOTAL = 189_093
num_batches  = -(-CHUNKS_TOTAL // optimal_batch_size)
eta_seconds  = num_batches * (SLEEP_SECONDS + 2)
eta_hours    = eta_seconds / 3600

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  AZURE QUOTA REPORT")
print("=" * 55)
print(f"  Region                 : {region}")
print(f"  Deployment             : {EMBED_MODEL}")
print()
print(f"  TPM limit              : {tpm_limit:>12,}")
print(f"  TPM remaining now      : {tpm_remaining:>12,}  ({100-tpm_used_pct:.1f}% free)")
print()
print(f"  RPM limit              : {rpm_limit:>12,}")
print(f"  RPM remaining now      : {rpm_remaining:>12,}  ({100-rpm_used_pct:.1f}% free)")
print("=" * 55)
print()
print("  RECOMMENDED SETTINGS")
print("─" * 55)
print(f"  AVG_TOKENS_PER_CHUNK   : {AVG_TOKENS_PER_CHUNK}  (1200-char chunks)")
print(f"  Safety margin          : {int(SAFETY_MARGIN*100)}%")
print(f"  Safe tokens/request    : {safe_tokens_per_req:,}")
print()
print(f"  ✅ MAX_BATCH_SIZE = {optimal_batch_size}")
print(f"  ✅ MIN_SLEEP      = {SLEEP_SECONDS}s")
print()
print(f"  ETA for 189K chunks    : {num_batches} batches × {SLEEP_SECONDS}s")
print(f"                           ≈ {eta_hours:.1f} hours")
print("=" * 55)
print()
print("  COPY THIS INTO config.py:")
print("─" * 55)
print(f"  MAX_BATCH_SIZE = {optimal_batch_size}")
print()
print("  COPY THIS INTO embedding.py constants:")
print("─" * 55)
print(f"  MIN_SLEEP = {SLEEP_SECONDS}.0")
print()

# ── Dump all rate-limit headers for reference ─────────────────────────────────
print("  RAW QUOTA HEADERS (all x-ratelimit-* from Azure):")
print("─" * 55)
for k, v in sorted(headers.items()):
    if "ratelimit" in k.lower() or "ms-region" in k.lower():
        print(f"  {k:<45}: {v}")
print("=" * 55)