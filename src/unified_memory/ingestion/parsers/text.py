"""
Plain text and markdown document parser.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO, List

from unified_memory.core.types import PageContent, SourceReference, SourceType
from unified_memory.ingestion.parsers.base import DocumentParser, ParsedDocument


class TextParser(DocumentParser):
    """
    Parser for plain text and markdown files.
    
    Treats the entire file as a single page.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md", ".markdown", ".rst", ".text"]
    
    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "text/plain",
            "text/markdown",
            "text/x-markdown",
            "text/x-rst",
        ]

    @property
    def default_source_type(self) -> SourceType:
        """Text documents are modeled as TEXT_BLOCK sources."""
        return SourceType.TEXT_BLOCK
    
    async def parse(
        self,
        source: BinaryIO,
        source_ref: SourceReference,
        document_id: str,
        encoding: str = "utf-8",
        **options: Any,
    ) -> ParsedDocument:
        """
        Parse a text file.
        
        Args:
            source: Binary stream of the text file
            source_ref: Source reference
            document_id: Document ID
            encoding: Text encoding (default: utf-8)
        """
        errors = []
        
        try:
            content = source.read()
            if isinstance(content, bytes):
                text = content.decode(encoding)
            else:
                text = content
        except UnicodeDecodeError as e:
            errors.append(f"Encoding error: {e}")
            # Try with error handling
            content = source.read() if hasattr(source, 'read') else content
            text = content.decode(encoding, errors="replace") if isinstance(content, bytes) else str(content)
        
        # Extract title from first line if it looks like a header
        title = None
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip()
            # Markdown header or all-caps title
            if first_line.startswith("#") or (first_line.isupper() and len(first_line) < 100):
                title = first_line.lstrip("#").strip()
        
        # Create single page content
        page = PageContent(
            page_number=1,
            document_id=document_id,
            text_blocks=[{"text": text}],
            full_text=text,
        )
        
        return ParsedDocument(
            document_id=document_id,
            source=source_ref,
            title=title,
            pages=[page],
            full_text=text,
            metadata={
                "encoding": encoding,
                "line_count": len(lines),
                "char_count": len(text),
            },
            parse_errors=errors,
        )
