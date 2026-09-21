import os
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import OpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"


class NVIDIAOpenAIEmbeddings(Embeddings):
    """LangChain embeddings adapter for NVIDIA's OpenAI-compatible API."""

    def __init__(self, batch_size: int = 50, client: Any | None = None):
        if client is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise RuntimeError("Set NVIDIA_API_KEY before using embeddings.")
            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

        self.client = client
        self.batch_size = batch_size

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=NVIDIA_EMBEDDING_MODEL,
                input=texts[start : start + self.batch_size],
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]
