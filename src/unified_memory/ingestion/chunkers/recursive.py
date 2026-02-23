"""
Recursive chunker implementation.

Designed for hierarchical documents (e.g., Markdown, Code).
"""

from __future__ import annotations

from typing import Any, List, Optional

from unified_memory.core.types import Chunk
from unified_memory.ingestion.parsers.base import ParsedDocument
from unified_memory.ingestion.chunkers.base import Chunker, ChunkingConfig


class RecursiveChunker(Chunker):
    """
    Recursively split text using a list of separators.
    
    Tries to split by the first separator. If chunks are too large,
    recursively splits by the next separator.
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        super().__init__(config)
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    @property
    def name(self) -> str:
        return "recursive"
    
    async def chunk(
        self,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
    ) -> List[Chunk]:
        """
        Split document recursively.
        """
        chunks = []
        chunk_idx = 0
        
        # We process page by page to preserve page numbers
        # although recursive chunking often ignores page boundaries.
        # Design decision: Concatenate pages? Or chunk per page?
        # Chunk per page prevents cross-page chunks but preserves page info.
        # Let's chunk per page for now as it's safer for provenance.
        
        for page in document.pages:
            text = page.full_text or "\n".join(b["text"] for b in page.text_blocks)
            
            if not text.strip():
                continue
            
            # Recursive splitting
            text_chunks = self._recursive_split(
                text, 
                self.separators,
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            
            for t in text_chunks:
                if not t.strip(): continue
                
                chunk = self._create_chunk(
                    text=t,
                    document=document,
                    chunk_index=chunk_idx,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    page_number=page.page_number
                )
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Recursively split text using the given separators.
        """
        final_chunks = []
        
        # 1. Base case: no separators left
        if not separators:
            # If no separators left and text is still too large, 
            # perform character-based splitting without overlap for safety.
            if len(text) > chunk_size:
                return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            return [text]
            
        separator = separators[0]
        next_separators = separators[1:]
        
        # 2. Split by current separator
        if separator == "":
            splits = list(text)  # split by character
        else:
            splits = text.split(separator)
            
        # 3. Process splits
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split_len = len(split)
            
            # If split is too big, recurse on it
            if split_len > chunk_size:
                # Flush current
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Recurse
                sub_chunks = self._recursive_split(
                    split, next_separators, chunk_size, chunk_overlap
                )
                final_chunks.extend(sub_chunks)
                continue
                
            # If adding would exceed size, flush
            if current_len + split_len + (len(separator) if current_chunk else 0) > chunk_size:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    
                    # Handle overlap: Keep some elements from current_chunk for next
                    # We take as many elements as fit within chunk_overlap
                    overlap_chunk = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) + (len(separator) if overlap_chunk else 0) <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_len += len(s) + (len(separator) if overlap_chunk > [s] else 0)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len
            
            current_chunk.append(split)
            current_len += split_len + (len(separator) if len(current_chunk) > 1 else 0)
            
        # Flush remaining
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        return final_chunks
