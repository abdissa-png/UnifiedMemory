"""
OpenAI Embedding Provider.

Uses the OpenAI API to generate text embeddings.
Requires the `openai` package.
"""

from __future__ import annotations

import base64
import io
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
        max_batch_size_text: int = 2048,
        max_batch_size_image: int = 20,
        supported_modalities: Optional[List[Modality]] = None,
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
        self._max_batch_size_text = max_batch_size_text
        self._max_batch_size_image = max_batch_size_image
        self._supported_modalities = supported_modalities or [Modality.TEXT]

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supported_modalities(self) -> List[Modality]:
        return self._supported_modalities

    @external_call()
    async def embed(
        self,
        content: Any,
        modality: Modality = Modality.TEXT,
    ) -> List[float]:
        self.validate_modality(modality)
        model_input = self._prepare_input(content, modality)
        response = await self._client.embeddings.create(
            input=[model_input],
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
        model_inputs = [self._prepare_input(content, modality) for content in contents]
        all_embeddings: List[List[float]] = [[] for _ in model_inputs]
        # Batch embedding for image modality is handled differently
        max_batch_size = self._max_batch_size_image if modality == Modality.IMAGE else self._max_batch_size_text
        for start in range(0, len(model_inputs), max_batch_size):
            batch = model_inputs[start : start + max_batch_size]
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
                dimensions=self._dimension,
            )
            for item in response.data:
                all_embeddings[start + item.index] = item.embedding
            self._record_usage(response.usage)

        return all_embeddings

    def _prepare_input(self, content: Any, modality: Modality) -> Any:
        if modality == Modality.TEXT:
            return content if isinstance(content, str) else str(content)
        if modality in (Modality.IMAGE, Modality.DOCUMENT):
            return self._prepare_image_input(content)
        # For modalities beyond image/text, pass through raw content for
        # compatible backends that implement custom semantics.
        return content

    def _prepare_image_input(self, content: Any) -> Any:
        if isinstance(content, dict):
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                return image_url
            return str(content)
        if isinstance(content, list):
            if (
                len(content) == 1
                and isinstance(content[0], dict)
                and isinstance(content[0].get("image_url"), str)
            ):
                return content[0]["image_url"]
            return str(content)
        if isinstance(content, bytes):
            encoded = base64.b64encode(content).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        if isinstance(content, str):
            value = content.strip()
            if (
                value.startswith("http://")
                or value.startswith("https://")
                or value.startswith("data:")
            ):
                return value
            return value
        if hasattr(content, "save"):
            buf = io.BytesIO()
            content.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        return str(content)

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
