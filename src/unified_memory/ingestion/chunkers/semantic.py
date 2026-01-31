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
        embedding_provider: EmbeddingProvider,
        config: Optional[ChunkingConfig] = None,
    ) -> None:
        super().__init__(config)
        self.embedding_provider = embedding_provider
        self.similarity_threshold = config.similarity_threshold if config else 0.5
    
    @property
    def name(self) -> str:
        return "semantic"
    
    async def chunk(
        self,
        document: ParsedDocument,
        namespace: str,
        embedding_model: str,
    ) -> List[Chunk]:
        """
        Split document into semantic chunks.
        """
        chunks = []
        chunk_idx = 0
        
        # Process page by page (or full text if preferred)
        # Semantic mapping works best on continuous text, so let's try full text
        # But we lose page mapping unless we track it carefully.
        # For simplicity MVP: Chunk each page semantically.
        
        for page in document.pages:
            text = "\n".join(b["text"] for b in page.text_blocks)
            if not text.strip():
                continue
            
            page_chunks = await self._chunk_semantics(text)
            
            for chunk_text in page_chunks:
                chunk = self._create_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_idx,
                    namespace=namespace,
                    embedding_model=embedding_model,
                    page_number=page.page_number
                )
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks

    async def _chunk_semantics(self, text: str) -> List[str]:
        """
        Split text based on semantic similarity.
        """
        # 1. Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return sentences
            
        # 2. Embed sentences
        embeddings = await self.embedding_provider.embed_batch(
            sentences, Modality.TEXT
        )
        
        # 3. Calculate cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[i+1])
            
            # Normalize
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            distance = 1 - similarity
            distances.append(distance)
        
        # 4. Determine split points
        # Split where distance > threshold (dissimilar)
        # Or use percentile based threshold for dynamic splitting
        
        # Simple thresholding
        config_threshold = 1 - self.similarity_threshold  # Convert sim to dist
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, distance in enumerate(distances):
            # distances[i] is between sentence[i] and sentence[i+1]
            if distance > config_threshold:
                # Split here
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                # Keep together
                current_chunk.append(sentences[i+1])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
