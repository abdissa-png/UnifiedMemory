import pytest
from pathlib import Path
from unified_memory.ingestion.parsers.registry import ParserRegistry
from unified_memory.ingestion.parsers.text import TextParser

def test_parser_registry_instances_are_isolated():
    """ParserRegistry is now a plain injectable class."""
    reg1 = ParserRegistry()
    reg2 = ParserRegistry()

    assert reg1 is not reg2

def test_parser_registry_registration():
    """Verify registration by extension and MIME."""
    registry = ParserRegistry()
    
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
