"""
Parser registry for document parsers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from unified_memory.ingestion.parsers.base import DocumentParser


class ParserRegistry:
    """
    Registry of available document parsers.
    """

    def __init__(self) -> None:
        self._ext_map: Dict[str, DocumentParser] = {}
        self._mime_map: Dict[str, DocumentParser] = {}
    
    def register(self, parser: DocumentParser) -> None:
        """Register a parser instance."""
        for ext in parser.supported_extensions:
            self._ext_map[ext.lower()] = parser
        for mime in parser.supported_mime_types:
            self._mime_map[mime.lower()] = parser
    
    def get_parser_for_file(
        self, 
        path: Path, 
        mime_type: Optional[str] = None
    ) -> Optional[DocumentParser]:
        """
        Find a suitable parser for the given file.
        
        Args:
            path: Path to the file
            mime_type: Optional MIME type
        """
        # 1. Try MIME type if provided
        if mime_type:
            parser = self._mime_map.get(mime_type.lower())
            if parser:
                return parser
        
        # 2. Try extension
        return self._ext_map.get(path.suffix.lower())

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._ext_map.clear()
        self._mime_map.clear()
