from typing import Optional
from openai import AzureOpenAI

client: Optional[AzureOpenAI] = None
client_emb: Optional[AzureOpenAI] = None