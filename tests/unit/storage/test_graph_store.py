import pytest
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.core.types import NodeType, EntityNode, GraphEdge

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
