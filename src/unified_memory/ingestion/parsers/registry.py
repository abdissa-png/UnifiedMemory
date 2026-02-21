"""
Parser Registry Singleton.

Central registry for document parsers, providing lookup by extension or MIME type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from unified_memory.ingestion.parsers.base import DocumentParser


class ParserRegistry:
    """
    Registry of available document parsers.
    
    Implemented as a singleton to avoid redundant registrations.
    """
    
    _instance: Optional[ParserRegistry] = None
    
    def __new__(cls) -> ParserRegistry:
        if cls._instance is None:
            cls._instance = super(ParserRegistry, cls).__new__(cls)
            cls._instance._ext_map = {}
            cls._instance._mime_map = {}
        return cls._instance
    
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


def get_parser_registry() -> ParserRegistry:
    """Get the global ParserRegistry singleton."""
    return ParserRegistry()
