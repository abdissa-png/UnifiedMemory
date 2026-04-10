"""
OpenAI Embedding Provider.

Uses the OpenAI API to generate text embeddings.
Requires the `openai` package.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from unified_memory.core.resilience import external_call
from unified_memory.core.types import Modality
from unified_memory.embeddings.base import EmbeddingProvider

from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider backed by the OpenAI embeddings API.

    Supports ``text-embedding-3-small``, ``text-embedding-3-large``,
    ``text-embedding-ada-002``, and any future models exposed via the
    same endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: Optional[str] = None,
        max_batch_size: int = 2048,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._dimension = dimension
        self._max_batch_size = max_batch_size

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.TEXT]

    @external_call()
    async def embed(
        self,
        content: Any,
        modality: Modality = Modality.TEXT,
    ) -> List[float]:
        self.validate_modality(modality)
        text = content if isinstance(content, str) else str(content)
        response = await self._client.embeddings.create(
            input=[text],
            model=self._model,
            dimensions=self._dimension,
        )
        self._record_usage(response.usage)
        return response.data[0].embedding

    @external_call()
    async def embed_batch(
        self,
        contents: List[Any],
        modality: Modality = Modality.TEXT,
    ) -> List[List[float]]:
        self.validate_modality(modality)
        texts = [c if isinstance(c, str) else str(c) for c in contents]
        all_embeddings: List[List[float]] = [[] for _ in texts]

        for start in range(0, len(texts), self._max_batch_size):
            batch = texts[start : start + self._max_batch_size]
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
                dimensions=self._dimension,
            )
            for item in response.data:
                all_embeddings[start + item.index] = item.embedding
            self._record_usage(response.usage)

        return all_embeddings

    def _record_usage(self, usage) -> None:
        """Extract token counts from CreateEmbeddingResponse.usage."""
        try:
            from unified_memory.observability.tracing import record_usage, UsageRecord

            record_usage(
                UsageRecord(
                    service="openai",
                    model=self._model,
                    operation="embedding",
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                )
            )
        except Exception as exc:
            log_event(logger, logging.DEBUG, "openai.embedding.usage.failed", error=str(exc))
