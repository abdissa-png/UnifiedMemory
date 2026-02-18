import pytest
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.core.types import NodeType, EntityNode, GraphEdge, make_entity_id

@pytest.fixture
def graph_store():
    return NetworkXGraphStore()

@pytest.mark.asyncio
async def test_node_operations(graph_store):
    node = EntityNode(
        id="e1",
        content="Entity 1",
        entity_name="E1",
        properties={"score": 0.9},
        node_type=NodeType.ENTITY
    )
    
    # Create
    nid = await graph_store.create_node(node, namespace="ns1")
    assert nid == "e1"
    
    # Get
    fetched = await graph_store.get_node("e1", namespace="ns1")
    assert fetched.id == "e1"
    assert fetched.content == "Entity 1"
    assert fetched.properties["score"] == 0.9
    
    # Get wrong namespace
    fetched = await graph_store.get_node("e1", namespace="ns2")
    assert fetched is None
    
    # Delete
    deleted = await graph_store.delete_node("e1", namespace="ns1")
    assert deleted is True
    fetched = await graph_store.get_node("e1", namespace="ns1")
    assert fetched is None

@pytest.mark.asyncio
async def test_edge_operations(graph_store):
    n1 = EntityNode(id="n1", content="Node 1", entity_name="Node 1", node_type=NodeType.ENTITY)
    n2 = EntityNode(id="n2", content="Node 2", entity_name="Node 2", node_type=NodeType.ENTITY)
    await graph_store.create_nodes_batch([n1, n2], namespace="ns1")
    
    edge = GraphEdge(
        source_id="n1",
        target_id="n2",
        relation="related_to",
        weight=0.5
    )
    
    # Create
    eid = await graph_store.create_edge(edge, namespace="ns1")
    assert eid is not None
    
    # Neighbors (out)
    neighbors = await graph_store.get_neighbors("n1", namespace="ns1", direction="out")
    assert len(neighbors) == 1
    assert neighbors[0].id == "n2"
    
    # Neighbors (in)
    neighbors = await graph_store.get_neighbors("n2", namespace="ns1", direction="in")
    assert len(neighbors) == 1
    assert neighbors[0].id == "n1"
    
    # Delete Edge
    count = await graph_store.delete_edges(source_id="n1", target_id="n2", namespace="ns1")
    assert count == 1
    neighbors = await graph_store.get_neighbors("n1", namespace="ns1", direction="out")
    assert len(neighbors) == 0

@pytest.mark.asyncio
async def test_pagerank_and_subgraph(graph_store):
    # A -> B -> C
    # A -> C
    nodes = [
        EntityNode(id="A", content="A", entity_name="A", node_type=NodeType.ENTITY),
        EntityNode(id="B", content="B", entity_name="B", node_type=NodeType.ENTITY),
        EntityNode(id="C", content="C", entity_name="C", node_type=NodeType.ENTITY)
    ]
    await graph_store.create_nodes_batch(nodes, namespace="ns1")
    
    edges = [
        GraphEdge(source_id="A", target_id="B", relation="rel"),
        GraphEdge(source_id="B", target_id="C", relation="rel"),
        GraphEdge(source_id="A", target_id="C", relation="rel")
    ]
    for e in edges:
        await graph_store.create_edge(e, namespace="ns1")
        
    # PPR from A
    scores = await graph_store.personalized_pagerank(["A"], namespace="ns1")
    # A should be highest (seed), then B/C
    assert scores[0][0] == "A"
    
    # Subgraph from A (hops=1 should give A, B, C)
    sub_nodes, sub_edges = await graph_store.get_subgraph(["A"], namespace="ns1", max_hops=1)
    node_ids = {n.id for n in sub_nodes}
    assert "A" in node_ids
    # B and C are connected to A, so they should return
    assert "B" in node_ids
    assert "C" in node_ids
    assert len(sub_edges) >= 2


@pytest.mark.asyncio
async def test_add_remove_namespace_node(graph_store):
    """Test split add/remove namespace methods for nodes."""
    node = EntityNode(
        id="e1", content="test", entity_name="E1", node_type=NodeType.ENTITY
    )
    await graph_store.create_node(node, namespace="ns1")

    # Add a second namespace
    ok = await graph_store.add_namespace_to_node("e1", "ns2")
    assert ok is True

    # Visible in both
    assert await graph_store.get_node("e1", namespace="ns1") is not None
    assert await graph_store.get_node("e1", namespace="ns2") is not None

    # Remove first namespace
    success, was_last = await graph_store.remove_namespace_from_node("e1", "ns1")
    assert success is True
    assert was_last is False
    assert await graph_store.get_node("e1", namespace="ns1") is None
    assert await graph_store.get_node("e1", namespace="ns2") is not None

    # Remove last namespace — node should be deleted
    success, was_last = await graph_store.remove_namespace_from_node("e1", "ns2")
    assert success is True
    assert was_last is True
    assert await graph_store.get_node("e1", namespace="ns2") is None


@pytest.mark.asyncio
async def test_add_remove_namespace_edge(graph_store):
    """Test split add/remove namespace methods for edges."""
    n1 = EntityNode(id="n1", content="N1", entity_name="N1", node_type=NodeType.ENTITY)
    n2 = EntityNode(id="n2", content="N2", entity_name="N2", node_type=NodeType.ENTITY)
    await graph_store.create_nodes_batch([n1, n2], namespace="ns1")

    edge = GraphEdge(
        id="edge1", source_id="n1", target_id="n2", relation="KNOWS"
    )
    await graph_store.create_edge(edge, namespace="ns1")

    ok = await graph_store.add_namespace_to_edge("edge1", "ns2")
    assert ok is True

    # Nonexistent edge
    ok = await graph_store.add_namespace_to_edge("nope", "ns2")
    assert ok is False

    success, was_last = await graph_store.remove_namespace_from_edge("edge1", "ns1")
    assert success is True
    assert was_last is False

    success, was_last = await graph_store.remove_namespace_from_edge("edge1", "ns2")
    assert success is True
    assert was_last is True


# ---------------------------------------------------------------------------
# C1: Round-trip persistence of entity_name, entity_type, and relation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_entity_fields_round_trip(graph_store):
    """entity_name and entity_type must survive a create→get round-trip."""
    node = EntityNode(
        id="ent1",
        content="Alice is a software engineer.",
        entity_name="Alice",
        entity_type="Person",
        node_type=NodeType.ENTITY,
    )
    await graph_store.create_node(node, namespace="ns1")

    fetched = await graph_store.get_node("ent1", namespace="ns1")
    assert fetched is not None
    assert isinstance(fetched, EntityNode)
    assert fetched.entity_name == "Alice"
    assert fetched.entity_type == "Person"
    # node_type must be preserved as the structural enum
    assert fetched.node_type == NodeType.ENTITY


@pytest.mark.asyncio
async def test_edge_relation_round_trip(graph_store):
    """The relation label must survive a create→internal-store round-trip."""
    n1 = EntityNode(id="a", content="A", entity_name="A", node_type=NodeType.ENTITY)
    n2 = EntityNode(id="b", content="B", entity_name="B", node_type=NodeType.ENTITY)
    await graph_store.create_nodes_batch([n1, n2], namespace="ns1")

    edge = GraphEdge(
        id="e-ab",
        source_id="a",
        target_id="b",
        relation="WORKS_FOR",
    )
    eid = await graph_store.create_edge(edge, namespace="ns1")
    assert eid is not None

    # Verify the relation is persisted in the underlying NetworkX graph.
    # The store uses a MultiDiGraph; edge data is keyed by edge UUID.
    edges_data = dict(graph_store._graph.get_edge_data("a", "b") or {})
    assert edges_data, "Expected at least one edge between 'a' and 'b'"
    relation_values = [d.get("relation") for d in edges_data.values()]
    assert "WORKS_FOR" in relation_values


@pytest.mark.asyncio
async def test_batch_entity_fields_round_trip(graph_store):
    """entity_name and entity_type must survive a create_nodes_batch→get round-trip."""
    nodes = [
        EntityNode(
            id="ent-batch-1",
            content="OpenAI",
            entity_name="OpenAI",
            entity_type="Organization",
            node_type=NodeType.ENTITY,
        ),
        EntityNode(
            id="ent-batch-2",
            content="Sam Altman",
            entity_name="Sam Altman",
            entity_type="Person",
            node_type=NodeType.ENTITY,
        ),
    ]
    await graph_store.create_nodes_batch(nodes, namespace="ns1")

    for node in nodes:
        fetched = await graph_store.get_node(node.id, namespace="ns1")
        assert fetched is not None
        assert isinstance(fetched, EntityNode)
        assert fetched.entity_name == node.entity_name
        assert fetched.entity_type == node.entity_type


# ---------------------------------------------------------------------------
# C2: make_entity_id collision behaviour
# ---------------------------------------------------------------------------

def test_make_entity_id_same_name_merges():
    """Entities with the same name map to the same ID regardless of type."""
    id1 = make_entity_id("Python", "tenant1")
    id2 = make_entity_id("Python", "tenant1")
    assert id1 == id2, "Same name/tenant must produce the same stable ID (merge by name)"


def test_make_entity_id_case_insensitive():
    """IDs are case-insensitive (normalised to lowercase)."""
    assert make_entity_id("Alice", "t1") == make_entity_id("alice", "t1")
    assert make_entity_id("ALICE", "t1") == make_entity_id("  alice  ", "t1")


def test_make_entity_id_different_tenants():
    """The same name in different tenants must produce different IDs."""
    assert make_entity_id("Python", "tenant1") != make_entity_id("Python", "tenant2")


# ---------------------------------------------------------------------------
# C3: node_type vs entity_type strict separation
# ---------------------------------------------------------------------------

def test_entity_node_type_is_always_entity():
    """EntityNode.node_type must always be NodeType.ENTITY, not the extractor label."""
    node = EntityNode(
        id="x",
        content="",
        entity_name="GPT-4",
        entity_type="Model",
        node_type=NodeType.ENTITY,
    )
    # Structural field
    assert node.node_type == NodeType.ENTITY
    # Domain label from extractor lives in entity_type, NOT node_type
    assert node.entity_type == "Model"
    assert node.entity_type != node.node_type.value


def test_entity_node_type_not_overwritten_by_entity_type():
    """Changing entity_type must not affect node_type."""
    node = EntityNode(
        id="y",
        content="",
        entity_name="London",
        entity_type="City",
        node_type=NodeType.ENTITY,
    )
    node.entity_type = "Location"
    assert node.node_type == NodeType.ENTITY  # structural field unchanged
