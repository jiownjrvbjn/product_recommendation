# init.py
import os
import logging
from typing import Optional
from openai import AzureOpenAI
import config.azure_sate as azure_state
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Globals (service singletons)
http_client: Optional[object] = None
http_client_emb: Optional[object] = None


def _require_env(var_name: str) -> str:
    """Ensure required env variable is present."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value

def initialize_services(*, http_client=None, http_client_emb=None, force: bool = False):
    if azure_state.client and azure_state.client_emb and not force:
        logger.info("Azure OpenAI services already initialized")
        return

    logger.info("Initializing Azure OpenAI services...")

    azure_state.client = AzureOpenAI(
        azure_endpoint=_require_env("ENDPOINT_URL"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        timeout=30.0,
        max_retries=3,
        http_client=http_client,
    )

    azure_state.client_emb = AzureOpenAI(
        azure_endpoint=_require_env("EMBEDDING_ENDPOINT"),
        api_key=_require_env("EMBEDDING_API_KEY"),
        api_version=os.getenv("EMBEDDING_API_VERSION", "2023-05-15"),
        timeout=30.0,
        max_retries=3,
        http_client=http_client_emb,
    )

    logger.info("Azure OpenAI services initialized successfully")