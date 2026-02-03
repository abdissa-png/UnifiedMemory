import pytest
import uuid
from unified_memory.core.types import (
    Entity, Relation, SourceLocation,
    GraphEdge, BaseGraphNode, EntityNode, NodeType, PassageNode, PageNode,
    source_locations_to_parallel_arrays, parallel_arrays_to_source_locations,
)

def test_entity_id_auto_generated():
    """Verify Entity auto-generates a UUID ID if not provided."""
    e = Entity(name="Alice")
    assert e.id is not None
    # Check if it's a valid UUID
    uuid.UUID(e.id)
    assert e.name == "Alice"

def test_entity_id_explicit():
    """Verify Entity uses provided ID."""
    custom_id = "custom-alice-123"
    e = Entity(id=custom_id, name="Alice")
    assert e.id == custom_id

def test_entity_add_source():
    """Verify add_source helper correctly updates source_locations."""
    e = Entity(name="Alice")
    e.add_source("doc1", "1")
    assert len(e.source_locations) == 1
    assert e.source_locations[0] == SourceLocation("doc1", 1)

    e.add_source("doc1", "2")
    assert len(e.source_locations) == 2

    # Dedup: adding the same source again should not duplicate
    e.add_source("doc1", "1")
    assert len(e.source_locations) == 2

def test_entity_source_locations():
    """source_locations tracks multiple documents and chunk indices."""
    e = Entity(name="Test")
    e.add_source("doc_a", "0")
    e.add_source("doc_b", "3")
    assert len(e.source_locations) == 2
    doc_ids = {loc.document_id for loc in e.source_locations}
    assert doc_ids == {"doc_a", "doc_b"}

def test_relation_id_generated():
    """Verify Relation also auto-generates ID."""
    r = Relation(subject_id="s1", predicate="knows", object_id="o1")
    assert r.id is not None
    uuid.UUID(r.id)

def test_relation_add_source_and_source_locations():
    """Verify Relation add_source updates source_locations."""
    r = Relation(subject_id="s1", predicate="knows", object_id="o1")
    r.add_source("doc1", "0")
    assert len(r.source_locations) == 1
    assert r.source_locations[0].document_id == "doc1"
    assert r.source_locations[0].chunk_index == 0
    r.add_source("doc1", "1")
    assert len(r.source_locations) == 2

def test_relation_source_locations_content():
    """source_locations on Relation tracks provenance correctly."""
    r = Relation(subject_id="s1", predicate="knows", object_id="o1")
    r.add_source("doc1", "0")
    r.add_source("doc1", "3")
    indices = {loc.chunk_index for loc in r.source_locations}
    assert indices == {0, 3}

def test_relation_to_from_dict():
    """Verify Relation to_dict/from_dict with source_locations."""
    r = Relation(subject_id="s1", predicate="knows", object_id="o1")
    r.add_source("doc1", "0")
    data = r.to_dict()
    assert "source_locations" in data
    assert data["source_locations"] == [{"document_id": "doc1", "chunk_index": 0}]
    r2 = Relation.from_dict(data)
    assert len(r2.source_locations) == 1
    assert r2.source_locations[0].document_id == "doc1"

def test_entity_to_from_dict():
    """Verify serialization preserves all fields."""
    e = Entity(name="Alice", entity_type="person")
    e.add_source("doc1", "1")

    data = e.to_dict()
    assert data["name"] == "Alice"
    assert data["source_locations"] == [{"document_id": "doc1", "chunk_index": 1}]
    assert "source_doc_ids" not in data
    assert "source_chunk_ids" not in data

    e2 = Entity.from_dict(data)
    assert e2.id == e.id
    assert e2.name == "Alice"
    assert len(e2.source_locations) == 1
    assert e2.source_locations[0].document_id == "doc1"

def test_graph_edge_source_locations():
    """GraphEdge uses source_locations for provenance."""
    loc = SourceLocation(document_id="doc1", chunk_index=2)
    edge = GraphEdge(
        source_id="a", target_id="b", relation="KNOWS",
        source_locations=[loc],
    )
    assert edge.source_locations[0].document_id == "doc1"
    assert edge.source_locations[0].chunk_index == 2

def test_base_graph_node_source_locations():
    """BaseGraphNode stores source_locations."""
    node = EntityNode(
        id="n1", node_type=NodeType.ENTITY, content="test",
        source_locations=[
            SourceLocation("doc1", 0),
            SourceLocation("doc2", 3),
        ],
        entity_name="Test",
    )
    assert len(node.source_locations) == 2
    doc_ids = {loc.document_id for loc in node.source_locations}
    assert doc_ids == {"doc1", "doc2"}

def test_parallel_array_conversion():
    """source_locations_to_parallel_arrays produces correct output."""
    locs = [
        SourceLocation("doc_a", 0),
        SourceLocation("doc_b", 5),
    ]
    result = source_locations_to_parallel_arrays(locs)
    assert result == {
        "source_doc_ids": ["doc_a", "doc_b"],
        "source_chunk_indices": [0, 5],
    }

def test_parallel_arrays_to_source_locations():
    """Reconstruct SourceLocation list from parallel arrays."""
    locs = parallel_arrays_to_source_locations(
        ["doc_a", "doc_b"], [0, 5]
    )
    assert len(locs) == 2
    assert locs[0] == SourceLocation("doc_a", 0)
    assert locs[1] == SourceLocation("doc_b", 5)

def test_passage_node_make_id():
    """PassageNode.make_id produces deterministic IDs."""
    pid = PassageNode.make_id("tenant1", "abc123")
    assert pid == "passage:tenant1:abc123"

def test_page_node_make_id():
    """PageNode.make_id produces deterministic IDs."""
    pid = PageNode.make_id("doc_a", 3)
    assert pid == "page:doc_a:3"
