from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    type: str # e.g. Person, Organization, Project, Concept
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """A relation extracted between entities."""
    source_entity: str
    target_entity: str
    relation_type: str  # e.g. WORKS_FOR, CREATED, MEMBER_OF

    description: Optional[str] = None
    confidence: float = 1.0
    weight: float = 1.0
    is_bidirectional: bool = False
    inverse_relation: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    # Optional entity type hints provided by the extractor.
    # These help disambiguate relations without changing the stable entity ID
    # (which remains name-only). They are stored as edge properties so that
    # downstream consumers can read them if needed.
    source_type: Optional[str] = None
    target_type: Optional[str] = None

    # Extractor-specific extras only
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Collection of extracted entities and relations."""

    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
