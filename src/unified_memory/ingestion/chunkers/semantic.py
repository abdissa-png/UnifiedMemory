"""
Semantic Chunker Implementation.

Splits text based on semantic similarity of consecutive sentences/blocks.
Requires an embedding provider.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Any, List, Optional, Tuple

from unified_memory.core.types import Chunk, Modality
from unified_memory.ingestion.parsers.base import ParsedDocument
from unified_memory.ingestion.chunkers.base import Chunker, ChunkingConfig
from unified_memory.embeddings.base import EmbeddingProvider


class SemanticChunker(Chunker):
    """
    Split text based on semantic similarity gaps.
    
    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Calculate cosine similarity between consecutive sentences
    4. Split where similarity drops below threshold (or local minima)
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        *,
        provider_registry: Optional["ProviderRegistry"] = None,
        namespace_manager: Optional["NamespaceManager"] = None,
        config: Optional[ChunkingConfig] = None,
    ) -> None:
        """
        Create a SemanticChunker.

        Two usage patterns are supported:
        - Tests / standalone: pass an explicit ``embedding_provider``.
        - Production: pass ``provider_registry`` and ``namespace_manager`` so
          the chunker can resolve the tenant's embedding provider from the
          namespace at call-time.
        """
        self._explicit_provider = embedding_provider
        self._provider_registry = provider_registry
        self._namespace_manager = namespace_manager
        self.similarity_threshold = config.similarity_threshold if config else 0.5
    
    @property
    def name(self) -> str:
        return "semantic"
    
    async def chunk(
        self,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """
        Split document into semantic chunks across pages.
        """
        cfg = config or ChunkingConfig(similarity_threshold=self.similarity_threshold)

        # Resolve embedding provider: prefer explicit instance (tests), otherwise
        # resolve via ProviderRegistry based on the namespace / tenant config.
        embedder = await self._resolve_embedding_provider(namespace)

        # 1. Concatenate all pages into continuous text with offset tracking
        full_text = ""
        page_map: List[Tuple[int, int, int]] = []  # (start, end, page_number)
        
        for page in document.pages:
            # Use pre-computed full_text if available, else join blocks
            page_text = page.full_text or "\n".join(b["text"] for b in page.text_blocks)
            if not page_text.strip():
                continue
                
            start_off = len(full_text)
            full_text += page_text + "\n"
            end_off = len(full_text)
            
            if page.page_number is not None:
                page_map.append((start_off, end_off, page.page_number))
        
        if not full_text:
            return []

        # 2. Perform semantic splitting on the full text
        # Returns List of (text, start_index, end_index)
        semantic_segments = await self._segment_semantics(full_text, cfg, embedder)
        
        chunks = []
        for i, (text, start, end) in enumerate(semantic_segments):
            # 3. Map segment back to source page
            # We use the page where the chunk STARTS.
            page_number = None
            for p_start, p_end, p_num in page_map:
                if p_start <= start < p_end:
                    page_number = p_num
                    break
            
            chunk = self._create_chunk(
                text=text,
                document=document,
                chunk_index=i,
                namespace=namespace,
                tenant_id=tenant_id,
                page_number=page_number,
                extra_metadata={
                    "start_char": start,
                    "end_char": end,
                    "is_semantic": True,
                },
                config=cfg,
            )
            chunks.append(chunk)
                
        return chunks

    async def _resolve_embedding_provider(self, namespace: str) -> EmbeddingProvider:
        """
        Determine which embedding provider to use for sentence embeddings.

        Resolution order:
        1. Explicit provider passed to constructor (used in unit tests).
        2. ProviderRegistry + NamespaceManager → TenantConfig.text_embedding.
        """
        if self._explicit_provider is not None:
            return self._explicit_provider

        if self._provider_registry is None or self._namespace_manager is None:
            raise ValueError(
                "SemanticChunker requires either an explicit embedding_provider "
                "or both provider_registry and namespace_manager to resolve one."
            )

        ns_config = await self._namespace_manager.get_config(namespace)
        if not ns_config:
            raise ValueError(f"Namespace not found for semantic chunking: {namespace}")

        tenant_config = await self._namespace_manager.get_tenant_config(ns_config.tenant_id)
        if not tenant_config:
            raise ValueError(
                f"Tenant config not found for semantic chunking (tenant={ns_config.tenant_id})."
            )

        model_cfg = tenant_config.text_embedding
        # Allow tests/configs that register providers keyed only by model id.
        embedder = self._provider_registry.resolve_embedding_provider(
            model_cfg.provider,
            model_cfg.model,
            fallback_key=model_cfg.model,
        )
        if embedder is None:
            raise ValueError(
                f"No embedding provider available for semantic chunking; "
                f"expected provider '{model_cfg.provider}' model '{model_cfg.model}'."
            )
        return embedder

    async def _segment_semantics(
        self, text: str, config: ChunkingConfig, embedding_provider: EmbeddingProvider
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into semantic segments with offsets.
        """
        # 1. Split into sentences and track offsets
        # Use regex to find sentences and their positions
        sentence_pattern = r'[^.!?]+[.!?]+(?:\s+|$)'
        matches = list(re.finditer(sentence_pattern, text))
        
        if not matches:
            # Fallback for text without punctuation
            return [(text, 0, len(text))]
        
        sentences = [m.group(0).strip() for m in matches]
        offsets = [(m.start(), m.end()) for m in matches]
            
        # 2. Embed sentences
        embeddings = await embedding_provider.embed_batch(
            sentences, Modality.TEXT
        )
        
        # 3. Calculate distances
        distances = []
        for i in range(len(embeddings) - 1):
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[i+1])
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            distances.append(1 - similarity)
        
        # 4. Determine split points
        config_threshold = 1 - config.similarity_threshold
        
        segments = []
        current_sentences = [sentences[0]]
        current_start = offsets[0][0]
        
        for i, distance in enumerate(distances):
            if distance > config_threshold:
                # Split
                segments.append((
                    " ".join(current_sentences),
                    current_start,
                    offsets[i][1]
                ))
                current_sentences = [sentences[i+1]]
                current_start = offsets[i+1][0]
            else:
                # Merge
                current_sentences.append(sentences[i+1])
        
        # Flush remaining
        if current_sentences:
            segments.append((
                " ".join(current_sentences),
                current_start,
                offsets[-1][1]
            ))
            
        return segments
