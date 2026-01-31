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
        embedding_model: str,
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
            text = "\n".join(b["text"] for b in page.text_blocks)
            
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
                    embedding_model=embedding_model,
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
            # If no separators left, we must hard split (character based) if still too large?
            # Or just return as is?
            # Let's just return as is for now, or character split.
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
            if current_len + split_len + len(separator) > chunk_size:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    
                    # Handle overlap (simplified)
                    # For strict overlap we'd need more complex logic here
                    current_chunk = []
                    current_len = 0
            
            current_chunk.append(split)
            current_len += split_len + len(separator)
            
        # Flush remaining
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        return final_chunks
