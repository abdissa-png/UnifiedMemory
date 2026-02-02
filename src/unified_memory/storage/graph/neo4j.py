"""
Neo4j Graph Store Backend.

Implements GraphStoreBackend using Neo4j and GDS (Graph Data Science) for PPR.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import asyncio
from neo4j import AsyncGraphDatabase

from unified_memory.storage.base import GraphStoreBackend, GraphNode, GraphEdge

logger = logging.getLogger(__name__)

class Neo4jGraphStore(GraphStoreBackend):
    """
    Neo4j implementation of GraphStoreBackend.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: Optional[Tuple[str, str]] = ("neo4j", "password"),
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)
        
    async def close(self):
        await self.driver.close()
        
    async def create_node(
        self,
        node: GraphNode,
        namespace: str,
    ) -> str:
        query = """
        MERGE (n:Entity {id: $id, namespace: $namespace})
        SET n += $properties, n.node_type = $node_type, n.content = $content
        RETURN n.id
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                id=node.id,
                namespace=namespace,
                node_type=str(node.node_type.value),
                content=node.content,
                properties=node.properties
            )
            record = await result.single()
            return record["n.id"]

    async def create_edge(
        self,
        edge: GraphEdge,
        namespace: str,
    ) -> str:
        # Dynamic relationship type is tricky in Cypher via parameters.
        # We must sanitize and inject. Assumes edge.relation is safe/validated.
        rel_type = edge.relation.upper().replace(" ", "_")
        
        query = f"""
        MATCH (s:Entity {{id: $source_id, namespace: $namespace}})
        MATCH (t:Entity {{id: $target_id, namespace: $namespace}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r += $properties, r.weight = $weight
        RETURN type(r)
        """
        async with self.driver.session() as session:
            await session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                namespace=namespace,
                weight=edge.weight,
                properties=edge.properties
            )
            return f"{edge.source_id}-{rel_type}->{edge.target_id}"

    async def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        namespace: str,
    ) -> List[str]:
        if not nodes:
            return []
            
        query = """
        UNWIND $batch as row
        MERGE (n:Entity {id: row.id, namespace: $namespace})
        SET n += row.properties, n.node_type = row.node_type, n.content = row.content
        RETURN n.id
        """
        batch_data = []
        for n in nodes:
            # Derive label from node_type or use entity_name for EntityNode
            label = str(n.node_type.value)
            if hasattr(n, 'entity_name') and n.entity_name:
                label = n.entity_name
            
            batch_data.append({
                "id": n.id,
                "node_type": str(n.node_type.value),
                "content": n.content,
                "properties": {**n.properties, "label": label}  # Store label in properties if needed
            })
        
        async with self.driver.session() as session:
            result = await session.run(query, batch=batch_data, namespace=namespace)
            records = await result.data()
            return [r["n.id"] for r in records]

    async def create_edges_batch(
        self,
        edges: List[GraphEdge],
        namespace: str,
    ) -> List[str]:
        if not edges:
            return []
            
        # Group by relation type for batching, as we can't parameterize rel type easily in UNWIND
        # Or use APOC if available. Assuming standard Cypher for compatibility.
        
        from collections import defaultdict
        edges_by_type = defaultdict(list)
        for e in edges:
            rel_type = e.relation.upper().replace(" ", "_")
            edges_by_type[rel_type].append(e)
            
        created_ids = []
        
        async with self.driver.session() as session:
            for rel_type, type_edges in edges_by_type.items():
                query = f"""
                UNWIND $batch as row
                MATCH (s:Entity {{id: row.source, namespace: $namespace}})
                MATCH (t:Entity {{id: row.target, namespace: $namespace}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += row.properties, r.weight = row.weight
                RETURN s.id, t.id
                """
                
                batch_data = [
                    {
                        "source": e.source,
                        "target": e.target,
                        "weight": e.weight,
                        "properties": e.properties
                    }
                    for e in type_edges
                ]
                
                await session.run(query, batch=batch_data, namespace=namespace)
                # Just mock return IDs for efficiency
                created_ids.extend([f"{e.source}-{rel_type}->{e.target}" for e in type_edges])
                
        return created_ids

    async def get_nodes_batch(
        self,
        node_ids: List[str],
        namespace: str,
    ) -> List[GraphNode]:
        query = """
        UNWIND $ids as node_id
        MATCH (n:Entity {id: node_id, namespace: $namespace})
        RETURN n
        """
        async with self.driver.session() as session:
            result = await session.run(query, ids=node_ids, namespace=namespace)
            records = await result.data()
            
            nodes = []
            for r in records:
                n = r["n"]
                props = dict(n)
                # Extract reserved fields
                nid = props.pop("id")
                # Handle label fallback
                nlabel = props.pop("node_type", "Entity")
                # Remove internal namespace prop
                props.pop("namespace", None)
                
                nodes.append(GraphNode(
                    id=nid,
                    label=nlabel,
                    properties=props,
                    content=props.pop("content", "")
                ))
            return nodes

    async def personalized_pagerank(
        self,
        seed_nodes: List[str],
        namespace: str,
        damping: float = 0.85,
        max_iterations: int = 100,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Run PPR using GDS.
        """
        # 1. Project sub-graph for namespace
        graph_name = f"graph_{namespace}"
        
        # Check if graph exists or project it
        # Simple ephemeral projection
        
        # NOTE: GDS projection is heavy. In production, maintain persistent projections or use algorithm on anonymous graph.
        # Anonymous graph example:
        
        query = """
        CALL gds.pageRank.stream(
          {
            nodeQuery: 'MATCH (n:Entity {namespace: $namespace}) RETURN id(n) as id',
            relationshipQuery: 'MATCH (n:Entity {namespace: $namespace})-[r]->(m:Entity {namespace: $namespace}) RETURN id(n) as source, id(m) as target, r.weight as weight',
            relationshipWeightProperty: 'weight',
            dampingFactor: $damping,
            maxIterations: $iterations,
            sourceNodes: $source_nodes
          }
        )
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id as id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        # Transform seed_nodes (string IDs) to internal Node IDs for GDS?
        # GDS sourceNodes can take internal IDs.
        # We need to lookup internal IDs first.
        
        async with self.driver.session() as session:
            # 1. Get internal IDs for seeds
            # Use elementalId() or stay with internal id() for now but fix the call
            id_query = "MATCH (n:Entity {namespace: $namespace}) WHERE n.id IN $seeds RETURN id(n) as internal_id"
            result = await session.run(id_query, seeds=seed_nodes, namespace=namespace)
            internal_ids = [record["internal_id"] async for record in result]
            
            if not internal_ids:
                return []
                
            # 2. Run PPR
            result = await session.run(
                query,
                namespace=namespace,
                damping=damping,
                iterations=max_iterations,
                source_nodes=internal_ids, # Pass list of integers
                top_k=top_k
            )
            
            ppr_results = []
            async for record in result:
                ppr_results.append((record["id"], record["score"]))
                
            return ppr_results

    # Implement other strict abstract methods with stubs or logic as needed
    async def get_node(self, node_id: str, namespace: str) -> Optional[GraphNode]:
        nodes = await self.get_nodes_batch([node_id], namespace)
        return nodes[0] if nodes else None

    async def get_neighbors(
        self,
        node_id: str,
        namespace: str,
        direction: str = "both",
        edge_types: Optional[List[str]] = None,
    ) -> List[GraphNode]:
        # Implementation omitted for brevity in this step, focusing on PPR
        return []

    async def query_nodes(self, filters, namespace=None, limit=100) -> List[Dict[str, Any]]:
        return []

    async def delete_node(self, node_id, namespace) -> bool:
        return True

    async def delete_edges(self, source_id=None, target_id=None, namespace=None) -> int:
        return 0

    async def get_subgraph(self, node_ids, namespace, max_hops=2) -> Tuple[List[GraphNode], List[GraphEdge]]:
        return [], []
