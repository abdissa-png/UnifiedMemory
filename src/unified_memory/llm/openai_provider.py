"""
OpenAI LLM Provider.

Uses langchain_openai's ChatOpenAI for text generation.
Requires the ``langchain-openai`` package (``pip install langchain-openai``).

Multimodal support
------------------
Pass ``supports_images=True`` when the chosen model supports vision input
(e.g. ``gpt-4o``, ``gpt-4-turbo``).  Text-only models (``gpt-4o-mini``,
``gpt-3.5-turbo``, …) must leave ``supports_images=False`` (the default).

When ``supports_images=True`` the ``generate_with_images`` method encodes
each image as a base64 data-URL and sends it alongside the prompt.
``base_url`` allows routing to OpenAI-compatible endpoints (Azure, local
proxies, Ollama, etc.) that use the same HTTP protocol.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, List, Optional

from unified_memory.core.resilience import external_call
from unified_memory.llm.base import BaseLLMProvider

from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """LLM provider backed by OpenAI's chat completions API via LangChain.

    Parameters
    ----------
    supports_images:
        When ``True``, ``generate_with_images`` is enabled.  Defaults to
        ``None`` which auto-detects based on the model name; set explicitly
        to ``False`` to disable image input even for vision-capable models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 128_000,
        base_url: Optional[str] = None,
        supports_images: Optional[bool] = False,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-openai' package is required for OpenAILLMProvider. "
                "Install it with: pip install langchain-openai"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)
        self._model = model
        self._max_tokens = max_tokens

        self.supports_images = supports_images

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @external_call()
    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        usage_callback: Optional[Any] = None,
    ) -> str:
        from langchain_core.messages import HumanMessage

        self._llm.temperature = temperature
        self._llm.max_tokens = max_output_tokens
        if stop_sequences:
            self._llm.stop = stop_sequences

        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        self._record_llm_usage(response, usage_callback)
        return str(response.content)

    async def generate_structured(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        usage_callback: Optional[Any] = None,
    ) -> str:
        return await self.generate(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            usage_callback=usage_callback,
        )

    @external_call()
    async def generate_with_images(
        self,
        prompt: str,
        images: List[bytes],
        max_output_tokens: int = 1024,
        usage_callback: Optional[Any] = None,
    ) -> str:
        """Generate a response from text + image inputs.

        Requires ``supports_images=True`` (or a vision-capable model).
        Images are base64-encoded and sent as data-URLs in the message
        content list, which is the OpenAI multimodal format.
        """
        if not self.supports_images:
            raise NotImplementedError(
                f"Model '{self._model}' was not configured for image input. "
                "Pass supports_images=True when constructing OpenAILLMProvider, "
                "or use a vision-capable model (e.g. gpt-4o)."
            )

        from langchain_core.messages import HumanMessage

        content: List[Any] = [{"type": "text", "text": prompt}]
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        self._llm.max_tokens = max_output_tokens
        response = await self._llm.ainvoke([HumanMessage(content=content)])
        self._record_llm_usage(response, usage_callback)
        return str(response.content)

    def _record_llm_usage(self, response, usage_callback: Optional[Any] = None) -> None:
        """Extract usage_metadata from AIMessage."""
        try:
            from unified_memory.observability.tracing import record_usage, UsageRecord

            usage = getattr(response, "usage_metadata", None)
            if not usage:
                return
            output_details = usage.get("output_token_details", {}) or {}
            input_details = usage.get("input_token_details", {}) or {}
            
            i_toks = usage.get("input_tokens", 0)
            o_toks = usage.get("output_tokens", 0)
            r_toks = usage.get("reasoning_tokens", 0)
            if usage_callback:
                try:
                    usage_callback(i_toks, o_toks, r_toks)
                except Exception as exc:
                    log_event(logger, logging.DEBUG, "openai.completion.usage.callback.failed", error=str(exc))

            try:
                record_usage(
                    UsageRecord(
                        service="openai",
                        model=self._model,
                        operation="completion",
                        input_tokens=i_toks,
                        output_tokens=o_toks,
                        reasoning_tokens=r_toks,
                        cache_read_tokens=input_details.get("cache_read", 0),
                        cache_creation_tokens=input_details.get("cache_creation", 0),
                    )
                )
            except Exception as exc:
                log_event(logger, logging.DEBUG, "openai.completion.usage.failed", error=str(exc))
        except Exception as exc:
            log_event(logger, logging.DEBUG, "openai.completion.usage.failed", error=str(exc))
