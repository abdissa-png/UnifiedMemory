import pytest
from pathlib import Path
from unified_memory.ingestion.parsers.registry import get_parser_registry, ParserRegistry
from unified_memory.ingestion.parsers.text import TextParser

def test_parser_registry_singleton():
    """Verify that get_parser_registry returns the same instance."""
    reg1 = get_parser_registry()
    reg2 = get_parser_registry()
    reg3 = ParserRegistry()
    
    assert reg1 is reg2
    assert reg1 is reg3

def test_parser_registry_registration():
    """Verify registration by extension and MIME."""
    registry = get_parser_registry()
    registry.clear()
    
    parser = TextParser()
    registry.register(parser)
    
    # Check extension
    assert registry.get_parser_for_file(Path("test.txt")) is parser
    assert registry.get_parser_for_file(Path("test.md")) is parser
    
    # Check MIME
    assert registry.get_parser_for_file(Path("rand.dat"), mime_type="text/plain") is parser
    assert registry.get_parser_for_file(Path("rand.dat"), mime_type="text/markdown") is parser

def test_parser_can_parse_mime():
    """Verify DocumentParser.can_parse handles MIME types."""
    parser = TextParser()
    
    # Path match
    assert parser.can_parse(Path("test.txt")) is True
    
    # MIME match (even if path doesn't match)
    assert parser.can_parse(Path("test.unknown"), mime_type="text/plain") is True
    
    # No match
    assert parser.can_parse(Path("test.unknown")) is False
    assert parser.can_parse(Path("test.txt"), mime_type="image/png") is True # Extension still matches
