"""
Base document parser interface.

Parsers extract structured content from raw documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, BinaryIO
from pathlib import Path

from unified_memory.core.types import PageContent, SourceReference, SourceType


@dataclass
class ParsedDocument:
    """
    Output of document parsing.
    
    Contains structured content extracted from a raw document,
    ready for chunking and further processing.
    """
    
    # Document identification
    document_id: str
    source: SourceReference
    
    # Extracted content
    title: Optional[str] = None
    pages: List[PageContent] = field(default_factory=list)
    
    # Full text (concatenated from pages if multi-page)
    full_text: str = ""
    
    # Document-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Parsing diagnostics
    parse_errors: List[str] = field(default_factory=list)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def has_errors(self) -> bool:
        return len(self.parse_errors) > 0


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    
    Implementations handle specific file formats:
    - TextParser: Plain text, markdown
    - PDFParser: PDF documents (with optional OCR)
    - HTMLParser: Web pages
    - etc.
    """
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        File extensions this parser can handle.
        
        Example: [".txt", ".md"] for text parser.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """
        MIME types this parser can handle.
        
        Example: ["text/plain", "text/markdown"] for text parser.
        """
        raise NotImplementedError
    
    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return path.suffix.lower() in self.supported_extensions
    
    @abstractmethod
    async def parse(
        self,
        source: BinaryIO,
        source_ref: SourceReference,
        document_id: str,
        **options: Any,
    ) -> ParsedDocument:
        """
        Parse a document from a binary stream.
        
        Args:
            source: Binary file-like object containing the document
            source_ref: Reference to the original source
            document_id: Unique identifier for this document
            **options: Parser-specific options
            
        Returns:
            ParsedDocument with extracted content
        """
        raise NotImplementedError
    
    async def parse_file(
        self,
        path: Path,
        document_id: Optional[str] = None,
        **options: Any,
    ) -> ParsedDocument:
        """
        Convenience method to parse a file from disk.
        
        Args:
            path: Path to the file
            document_id: Optional ID, defaults to filename
            **options: Parser-specific options
        """
        import uuid
        
        doc_id = document_id or str(uuid.uuid4())
        source_ref = SourceReference(
            source_id=doc_id,
            source_type=SourceType.TEXT_BLOCK,
        )
        
        with open(path, "rb") as f:
            return await self.parse(f, source_ref, doc_id, **options)
