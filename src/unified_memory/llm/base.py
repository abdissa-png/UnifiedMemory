"""
LLM Provider Base.

Defines the abstract base class for LLM providers used by extractors
and answer generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers.

    Implementors must override ``model_id``, ``max_tokens``, ``generate``,
    and ``generate_structured``.

    Multimodal (image-input) support is opt-in: override
    ``generate_with_images`` **and** set ``supports_images = True`` so that
    callers can check capability without catching ``NotImplementedError``.
    """

    # Subclasses that support image input must set this to True.
    supports_images: bool = False

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate with temperature=0 for deterministic structured output."""
        ...

    async def generate_with_images(
        self,
        prompt: str,
        images: List[bytes],
        max_output_tokens: int = 1024,
    ) -> str:
        """Generate a response using both text and image inputs.

        Not supported by default.  Subclasses that handle multimodal input
        must override this method **and** set ``supports_images = True``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support image input. "
            "Use a vision-capable model (e.g. gpt-4o) and set supports_images=True."
        )
