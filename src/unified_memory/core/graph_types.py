"""
Graph type exports extracted from ``core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import uuid

from .enums import NodeType
from .source_types import SourceLocation

@dataclass
class BaseGraphNode:
    """
    Base class for all graph nodes.

    Contains only fields common to ALL node types.
    Specific node types extend this with their own fields.
    """

    id: str
    node_type: NodeType
    content: str

    # Canonical provenance
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Namespace for isolation
    namespace: str = "default"

    # Flexible properties for extension
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode(BaseGraphNode):
    """Graph node representing an extracted entity."""

    # Entity-specific: edge counts for graph analytics
    incoming_edge_count: int = 0
    outgoing_edge_count: int = 0

    # Reference to entity type/name
    entity_type: str = "entity"
    entity_name: str = ""

    def __post_init__(self) -> None:
        self.node_type = NodeType.ENTITY


@dataclass
class PassageNode(BaseGraphNode):
    """Graph node representing a text passage/chunk."""

    page_number: Optional[int] = None

    # Entities mentioned in this passage
    mentioned_entity_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = NodeType.PASSAGE

    @staticmethod
    def make_id(tenant_id: str, content_hash: str) -> str:
        return f"passage:{tenant_id}:{content_hash}"


@dataclass
class PageNode(BaseGraphNode):
    """
    Graph node representing a document page (MMKG style).

    NOTE: Visual embeddings live in the vector store (page_images collection),
    not in the graph DB.
    """

    # Page-specific fields
    page_number: int = 0
    document_id: str = ""
    text_summary: str = ""

    # Entities and relations extracted from this page
    entity_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = NodeType.PAGE

    @staticmethod
    def make_id(document_id: str, page_number: int) -> str:
        return f"page:{document_id}:{page_number}"


# Type alias for any graph node (use for functions that accept any node type)
GraphNode = Union[EntityNode, PassageNode, PageNode]


@dataclass
class GraphEdge:
    """Graph edge - UNIFIED with direction semantics."""

    source_id: str  # FROM node
    target_id: str  # TO node
    relation: str
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    weight: float = 1.0

    # Direction semantics
    is_bidirectional: bool = False
    inverse_relation: Optional[str] = None

    # Optional denormalised names for embedding / display
    source_entity_name: str = ""
    target_entity_name: str = ""

    # Canonical provenance
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Namespace
    namespace: str = "default"

    properties: Dict[str, Any] = field(default_factory=dict)

__all__ = [
    "BaseGraphNode",
    "EntityNode",
    "GraphEdge",
    "GraphNode",
    "NodeType",
    "PageNode",
    "PassageNode",
]
