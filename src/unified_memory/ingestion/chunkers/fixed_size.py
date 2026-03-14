"""
Fixed-size chunker implementation.

Splits text into chunks of approximately equal size with configurable overlap.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from unified_memory.core.types import Chunk
from unified_memory.ingestion.parsers.base import ParsedDocument
from unified_memory.ingestion.chunkers.base import Chunker, ChunkingConfig


class FixedSizeChunker(Chunker):
    """
    Split documents into fixed-size chunks with overlap.
    
    Uses character count as the size metric.
    Optionally respects sentence boundaries.
    """
    
    @property
    def name(self) -> str:
        return "fixed_size"
    
    async def chunk(
        self,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """
        Split document into fixed-size chunks.
        """
        cfg = config or ChunkingConfig()

        all_chunks = []
        chunk_index_offset = 0
        
        # Guard against infinite loops
        if cfg.chunk_overlap >= cfg.chunk_size:
            raise ValueError(
                f"Chunk overlap ({cfg.chunk_overlap}) must be less than "
                f"chunk size ({cfg.chunk_size}) to avoid infinite loops."
            )
        
        for page in document.pages:
            text = page.full_text
            if not text.strip():
                continue
                
            if cfg.respect_sentence_boundaries:
                page_chunks = self._chunk_by_sentences(
                    text,
                    document,
                    namespace,
                    tenant_id,
                    page.page_number,
                    chunk_index_offset,
                    cfg,
                )
            else:
                page_chunks = self._chunk_by_characters(
                    text,
                    document,
                    namespace,
                    tenant_id,
                    page.page_number,
                    chunk_index_offset,
                    cfg,
                )
            
            all_chunks.extend(page_chunks)
            chunk_index_offset += len(page_chunks)
        
        return all_chunks
    
    def _chunk_by_characters(
        self,
        text: str,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
        page_number: Optional[int] = None,
        chunk_index_offset: int = 0,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """Simple character-based chunking."""
        cfg = config or ChunkingConfig()

        chunks = []
        chunk_size = cfg.chunk_size
        overlap = cfg.chunk_overlap
        
        start = 0
        chunk_index = chunk_index_offset
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        document=document,
                        chunk_index=chunk_index,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        page_number=page_number,
                        config=cfg,
                    )
                )
                chunk_index += 1
            
            # If we've reached or passed the end, we're done
            if end >= len(text):
                break
                
            # Move start with overlap
            start = end - overlap
            if start <= (end - chunk_size):
                # Prevent infinite loop
                start = end
        
        return chunks
    
    def _chunk_by_sentences(
        self,
        text: str,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
        page_number: Optional[int] = None,
        chunk_index_offset: int = 0,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """Chunk while respecting sentence boundaries."""
        cfg = config or ChunkingConfig()

        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = chunk_index_offset
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_len = len(sentence)
            
            # If single sentence exceeds max size, split it
            if sentence_len > cfg.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            text=chunk_text,
                            document=document,
                            chunk_index=chunk_index,
                            namespace=namespace,
                            tenant_id=tenant_id,
                            page_number=page_number,
                            config=cfg,
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by characters
                for i in range(0, sentence_len, cfg.chunk_size - cfg.chunk_overlap):
                    part = sentence[i : i + cfg.chunk_size]
                    if part.strip():
                        chunks.append(
                            self._create_chunk(
                                text=part,
                                document=document,
                                chunk_index=chunk_index,
                                namespace=namespace,
                                tenant_id=tenant_id,
                                page_number=page_number,
                                config=cfg,
                            )
                        )
                        chunk_index += 1
                continue
            
            # Check if adding this sentence would exceed limit
            if current_length + sentence_len + 1 > cfg.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        document=document,
                        chunk_index=chunk_index,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        page_number=page_number
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                # Take last few sentences for overlap
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) + 1 <= cfg.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = " ".join(overlap_sentences)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1
        
        # Flush remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                self._create_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    page_number=page_number,
                    config=cfg,
                )
            )
        
        return chunks
