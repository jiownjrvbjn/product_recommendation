# config.py
import os
import sys
import warnings
from dotenv import load_dotenv
from cachetools import LRUCache

# ===========================
# Environment
# ===========================
load_dotenv()
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    
# ============================
# Global Settings
# ============================
lancedb_client = None
last_processed_hashes = {}
answer_cache = LRUCache(maxsize=100)

# ===========================
# Paths
# ===========================
LANCE_PATH = os.getenv("LANCE_PATH", "lancedb_db")
COLLECTION_NAME_PREFIX = os.getenv("COLLECTION_NAME", "rag_collection")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ===========================
# Azure OpenAI
# ===========================
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
deployment_emb = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBED_MODEL = deployment_emb

# ===========================
# CSV Locations
# ===========================
CSV_FILES = {
    "sales_data1": os.getenv("CSV_Sales_data1"),
    "sales_data2": os.getenv("CSV_Sales_data2"),
    "callaverage_data1": os.getenv("CSV_CallAverage_data1"),
    "callaverage_data2": os.getenv("CSV_CallAverage_data2"),
    "coverage_data1": os.getenv("CSV_Coverage_data1"),
    "coverage_data2": os.getenv("CSV_Coverage_data2")
}

# ===========================
# RAG Limits
# ===========================
# Azure quota (confirmed via check_azure_quota.py — South India region):
#   TPM limit : 150,000   (hard ceiling)
#   RPM limit : 0         (unlimited — only token budget applies)
#
# Optimal formula (RPM=0 means only TPM matters):
#   tokens per request = TPM * 0.80 = 120,000
#   batch_size         = 120,000 / 300 tokens_per_chunk = 400
#   sleep              = 60 / (TPM / tokens_per_request) = 60 / 1.25 = 48s
#
# Throughput ceiling: 150,000 / 300 = 500 chunks/min (physics — cannot exceed this)
# ETA for 189K chunks: 473 batches × ~49.5s ≈ 6.5 hours
MAX_BATCH_SIZE = 400     
MAX_BATCH_TOKENS = 100_000
MAX_DOC_TOKENS = 8_000
MAX_ROWS = 600_000

# ===========================
# Performance
# ===========================
MAX_CACHE_SIZE = 100
MAX_EMBEDDING_WORKERS = 10
USE_PARALLEL_EMBEDDING = True
