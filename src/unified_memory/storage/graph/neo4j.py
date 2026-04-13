"""
Neo4j Graph Store Backend.

Implements GraphStoreBackend using Neo4j and GDS (Graph Data Science) for PPR.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import asyncio
from collections import defaultdict

from neo4j import AsyncGraphDatabase

from unified_memory.core.config import DEFAULT_NEO4J_URI
from unified_memory.storage.base import GraphStoreBackend, GraphNode, GraphEdge
from unified_memory.core.types import (
    EntityNode,
    PassageNode,
    PageNode,
    NodeType,
    SourceLocation,
    source_locations_to_parallel_arrays,
    parallel_arrays_to_source_locations,
)

logger = logging.getLogger(__name__)

class Neo4jGraphStore(GraphStoreBackend):
    """
    Neo4j implementation of GraphStoreBackend.
    """
    
    def __init__(
        self,
        uri: str = DEFAULT_NEO4J_URI,
        auth: Optional[Tuple[str, str]] = ("neo4j", "password"),
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)

    async def close(self):
        await self.driver.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def create_node(
        self,
        node: GraphNode,
        namespace: str,
    ) -> str:
        query = """
        MERGE (n:Entity {id: $id})
        ON CREATE SET
            n += $properties,
            n.node_type = $node_type,
            n.content = $content,
            n.namespaces = [$namespace],
            n.source_doc_ids = $source_doc_ids,
            n.source_chunk_indices = $source_chunk_indices
        ON MATCH SET
            n += $properties,
            n.node_type = $node_type,
            n.content = $content,
            n.namespaces = CASE
                WHEN NOT $namespace IN n.namespaces THEN n.namespaces + $namespace
                ELSE n.namespaces
            END,
            n.source_doc_ids = n.source_doc_ids + $source_doc_ids,
            n.source_chunk_indices = n.source_chunk_indices + $source_chunk_indices
        RETURN n.id AS id
        """
        provenance = source_locations_to_parallel_arrays(
            getattr(node, "source_locations", [])
        )
        async with self.driver.session() as session:
            result = await session.run(
                query,
                id=node.id,
                namespace=namespace,
                node_type=str(node.node_type.value),
                content=node.content,
                properties=node.properties,
                source_doc_ids=provenance["source_doc_ids"],
                source_chunk_indices=provenance["source_chunk_indices"],
            )
            record = await result.single()
            return record["id"]

    async def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        namespace: str,
    ) -> List[str]:
        if not nodes:
            return []

        query = """
        UNWIND $batch AS row
        MERGE (n:Entity {id: row.id})
        ON CREATE SET
            n += row.properties,
            n.node_type = row.node_type,
            n.content = row.content,
            n.namespaces = [$namespace],
            n.source_doc_ids = row.source_doc_ids,
            n.source_chunk_indices = row.source_chunk_indices
        ON MATCH SET
            n += row.properties,
            n.node_type = row.node_type,
            n.content = row.content,
            n.namespaces = CASE
                WHEN NOT $namespace IN n.namespaces THEN n.namespaces + $namespace
                ELSE n.namespaces
            END,
            n.source_doc_ids = n.source_doc_ids + row.source_doc_ids,
            n.source_chunk_indices = n.source_chunk_indices + row.source_chunk_indices
        RETURN n.id AS id
        """
        batch_data = []
        for n in nodes:
            label = str(n.node_type.value)
            if hasattr(n, "entity_name") and n.entity_name:
                label = n.entity_name
            provenance = source_locations_to_parallel_arrays(
                getattr(n, "source_locations", [])
            )
            batch_data.append(
                {
                    "id": n.id,
                    "node_type": str(n.node_type.value),
                    "content": n.content,
                    "properties": {**n.properties, "label": label},
                    "source_doc_ids": provenance["source_doc_ids"],
                    "source_chunk_indices": provenance["source_chunk_indices"],
                }
            )

        async with self.driver.session() as session:
            result = await session.run(query, batch=batch_data, namespace=namespace)
            records = await result.data()
            return [r["id"] for r in records]

    async def get_node(self, node_id: str, namespace: str) -> Optional[GraphNode]:
        nodes = await self.get_nodes_batch([node_id], namespace)
        return nodes[0] if nodes else None

    async def get_nodes_batch(
        self,
        node_ids: List[str],
        namespace: str,
    ) -> List[GraphNode]:
        query = """
        UNWIND $ids AS node_id
        MATCH (n:Entity {id: node_id})
        WHERE $namespace IN n.namespaces
        RETURN n
        """
        async with self.driver.session() as session:
            result = await session.run(query, ids=node_ids, namespace=namespace)
            records = await result.data()

            nodes = []
            for r in records:
                node_data = dict(r["n"])
                nodes.append(self._node_from_record(node_data, namespace))
            return nodes

    async def delete_node(self, node_id: str, namespace: str) -> bool:
        query = """
        MATCH (n:Entity {id: $id})
        WHERE $namespace IN n.namespaces
        WITH n, [x IN n.namespaces WHERE x <> $namespace] AS remaining
        FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
            DETACH DELETE n
        )
        FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
            SET n.namespaces = remaining
        )
        RETURN size(remaining) = 0 AS deleted
        """
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id, namespace=namespace)
            record = await result.single()
            return bool(record and record["deleted"])

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def create_edge(
        self,
        edge: GraphEdge,
        namespace: str,
    ) -> str:
        rel_type = edge.relation
        provenance = source_locations_to_parallel_arrays(
            getattr(edge, "source_locations", [])
        )

        query = f"""
        MATCH (s:Entity {{id: $source_id}})
        MATCH (t:Entity {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET
            r += $properties,
            r.id = $edge_id,
            r.weight = $weight,
            r.is_bidirectional = $is_bidirectional,
            r.inverse_relation = $inverse_relation,
            r.namespaces = [$namespace],
            r.source_doc_ids = $source_doc_ids,
            r.source_chunk_indices = $source_chunk_indices
        ON MATCH SET
            r += $properties,
            r.weight = $weight,
            r.namespaces = CASE
                WHEN NOT $namespace IN r.namespaces THEN r.namespaces + $namespace
                ELSE r.namespaces
            END,
            r.source_doc_ids = r.source_doc_ids + $source_doc_ids,
            r.source_chunk_indices = r.source_chunk_indices + $source_chunk_indices
        RETURN type(r) AS rel_type
        """
        async with self.driver.session() as session:
            await session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                namespace=namespace,
                edge_id=edge.id,
                weight=edge.weight,
                is_bidirectional=edge.is_bidirectional,
                inverse_relation=edge.inverse_relation,
                properties=edge.properties,
                source_doc_ids=provenance["source_doc_ids"],
                source_chunk_indices=provenance["source_chunk_indices"],
            )

        # Create inverse edge for bidirectional relations
        if edge.is_bidirectional and edge.inverse_relation:
            inv_query = f"""
            MATCH (s:Entity {{id: $target_id}})
            MATCH (t:Entity {{id: $source_id}})
            MERGE (s)-[r:{edge.inverse_relation}]->(t)
            ON CREATE SET
                r.id = $edge_id + '_inv',
                r.weight = $weight,
                r.is_bidirectional = true,
                r.namespaces = [$namespace],
                r.source_doc_ids = $source_doc_ids,
                r.source_chunk_indices = $source_chunk_indices
            ON MATCH SET
                r.weight = $weight,
                r.namespaces = CASE
                    WHEN NOT $namespace IN r.namespaces THEN r.namespaces + $namespace
                    ELSE r.namespaces
                END,
                r.source_doc_ids = r.source_doc_ids + $source_doc_ids,
                r.source_chunk_indices = r.source_chunk_indices + $source_chunk_indices
            RETURN type(r) AS rel_type
            """
            async with self.driver.session() as session:
                await session.run(
                    inv_query,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    namespace=namespace,
                    edge_id=edge.id,
                    weight=edge.weight,
                    source_doc_ids=provenance["source_doc_ids"],
                    source_chunk_indices=provenance["source_chunk_indices"],
                )

        return edge.id

    async def create_edges_batch(
        self,
        edges: List[GraphEdge],
        namespace: str,
    ) -> List[str]:
        if not edges:
            return []

        # NOTE: Cypher relationship types (e.g. `:KNOWS`) are part of the query
        # syntax and cannot be parameterized per row (you cannot do `r:$type`).
        # With the current data model (native Neo4j relationship types), batching
        # across multiple relation types therefore requires grouping by `edge.relation`
        # and issuing one UNWIND query per type.
        edges_by_type: Dict[str, List[GraphEdge]] = defaultdict(list)
        for e in edges:
            edges_by_type[e.relation].append(e)

        created_ids: List[str] = []

        async with self.driver.session() as session:
            for rel_type, type_edges in edges_by_type.items():
                query = f"""
                UNWIND $batch AS row
                MATCH (s:Entity {{id: row.source}})
                MATCH (t:Entity {{id: row.target}})
                MERGE (s)-[r:{rel_type}]->(t)
                ON CREATE SET
                    r += row.properties,
                    r.id = row.edge_id,
                    r.weight = row.weight,
                    r.is_bidirectional = row.is_bidirectional,
                    r.inverse_relation = row.inverse_relation,
                    r.namespaces = [$namespace],
                    r.source_doc_ids = row.source_doc_ids,
                    r.source_chunk_indices = row.source_chunk_indices
                ON MATCH SET
                    r += row.properties,
                    r.weight = row.weight,
                    r.namespaces = CASE
                        WHEN NOT $namespace IN r.namespaces THEN r.namespaces + $namespace
                        ELSE r.namespaces
                    END,
                    r.source_doc_ids = r.source_doc_ids + row.source_doc_ids,
                    r.source_chunk_indices = r.source_chunk_indices + row.source_chunk_indices
                RETURN s.id AS source_id, t.id AS target_id
                """

                batch_data = []
                for e in type_edges:
                    provenance = source_locations_to_parallel_arrays(
                        getattr(e, "source_locations", [])
                    )
                    batch_data.append(
                        {
                            "source": e.source_id,
                            "target": e.target_id,
                            "edge_id": e.id,
                            "weight": e.weight,
                            "is_bidirectional": e.is_bidirectional,
                            "inverse_relation": e.inverse_relation,
                            "properties": e.properties,
                            "source_doc_ids": provenance["source_doc_ids"],
                            "source_chunk_indices": provenance["source_chunk_indices"],
                        }
                    )

                await session.run(query, batch=batch_data, namespace=namespace)
                created_ids.extend(e.id for e in type_edges)

        return created_ids

    async def delete_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        where_clauses = []
        params: Dict[str, Any] = {}

        if source_id:
            where_clauses.append("s.id = $source_id")
            params["source_id"] = source_id
        if target_id:
            where_clauses.append("t.id = $target_id")
            params["target_id"] = target_id
        if namespace:
            where_clauses.append("$namespace IN r.namespaces")
            params["namespace"] = namespace

        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (s:Entity)-[r]->(t:Entity)
        {where_str}
        WITH r, [x IN r.namespaces WHERE x <> $namespace] AS remaining
        FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
            DELETE r
        )
        FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
            SET r.namespaces = remaining
        )
        RETURN count(r) AS affected
        """
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return int(record["affected"]) if record else 0

    # ------------------------------------------------------------------
    # Namespace operations (split)
    # ------------------------------------------------------------------

    async def add_namespace_to_node(
        self,
        node_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        query = """
        MATCH (n:Entity {id: $id})
        SET n.namespaces = CASE
            WHEN NOT $namespace IN n.namespaces THEN n.namespaces + $namespace
            ELSE n.namespaces
        END,
        n.source_doc_ids = CASE
            WHEN $document_id IS NOT NULL AND NOT $document_id IN n.source_doc_ids THEN n.source_doc_ids + $document_id
            ELSE n.source_doc_ids
        END
        RETURN n.id AS id
        """
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id, namespace=namespace, document_id=document_id)
            record = await result.single()
            return record is not None

    async def add_namespace_to_edge(
        self,
        edge_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        query = """
        MATCH ()-[r]->()
        WHERE r.id = $edge_id
        SET r.namespaces = CASE
            WHEN NOT $namespace IN r.namespaces THEN r.namespaces + $namespace
            ELSE r.namespaces
        END,
        r.source_doc_ids = CASE
            WHEN $document_id IS NOT NULL AND NOT $document_id IN r.source_doc_ids THEN r.source_doc_ids + $document_id
            ELSE r.source_doc_ids
        END
        RETURN r.id AS id
        """
        async with self.driver.session() as session:
            result = await session.run(query, edge_id=edge_id, namespace=namespace, document_id=document_id)
            record = await result.single()
            return record is not None

    async def add_namespace(self, id: str, namespace: str, document_id: Optional[str] = None) -> bool:
        if await self.add_namespace_to_node(id, namespace, document_id):
            return True
        return await self.add_namespace_to_edge(id, namespace, document_id)

    async def remove_namespace_from_node(
        self,
        node_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        query = """
        MATCH (n:Entity {id: $id})
        WHERE $namespace IN n.namespaces
        WITH n, [x IN n.namespaces WHERE x <> $namespace] AS remaining
        FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
            DETACH DELETE n
        )
        FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
            SET n.namespaces = remaining
        )
        RETURN size(remaining) = 0 AS was_last
        """
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id, namespace=namespace)
            record = await result.single()
            if record is None:
                return False, False
            return True, bool(record["was_last"])

    async def remove_namespace_from_edge(
        self,
        edge_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        query = """
        MATCH ()-[r]->()
        WHERE r.id = $edge_id AND $namespace IN r.namespaces
        WITH r, [x IN r.namespaces WHERE x <> $namespace] AS remaining
        FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
            DELETE r
        )
        FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
            SET r.namespaces = remaining
        )
        RETURN size(remaining) = 0 AS was_last
        """
        async with self.driver.session() as session:
            result = await session.run(query, edge_id=edge_id, namespace=namespace)
            record = await result.single()
            if record is None:
                return False, False
            return True, bool(record["was_last"])

    async def remove_namespace(
        self, id: str, namespace: str
    ) -> Tuple[bool, bool]:
        success, was_last = await self.remove_namespace_from_node(id, namespace)
        if success:
            return success, was_last
        return await self.remove_namespace_from_edge(id, namespace)

    async def remove_document_reference(
        self,
        id: str,
        document_id: str,
    ) -> List[str]:
        """Remove *document_id* from ``source_doc_ids`` /
        ``source_chunk_indices`` on a node or edge.
        Returns remaining ``source_doc_ids``."""
        # Try node first
        query_node = """
        MATCH (n:Entity {id: $id})
        WITH n,
             [i IN range(0, size(n.source_doc_ids)-1)
              WHERE n.source_doc_ids[i] <> $doc_id] AS keep
        SET n.source_doc_ids = [i IN keep | n.source_doc_ids[i]],
            n.source_chunk_indices = [i IN keep | n.source_chunk_indices[i]]
        RETURN n.source_doc_ids AS remaining
        """
        async with self.driver.session() as session:
            result = await session.run(query_node, id=id, doc_id=document_id)
            record = await result.single()
            if record is not None:
                return list(record["remaining"] or [])

        # Try edge
        query_edge = """
        MATCH ()-[r]->()
        WHERE r.id = $id
        WITH r,
             [i IN range(0, size(r.source_doc_ids)-1)
              WHERE r.source_doc_ids[i] <> $doc_id] AS keep
        SET r.source_doc_ids = [i IN keep | r.source_doc_ids[i]],
            r.source_chunk_indices = [i IN keep | r.source_chunk_indices[i]]
        RETURN r.source_doc_ids AS remaining
        """
        async with self.driver.session() as session:
            result = await session.run(query_edge, id=id, doc_id=document_id)
            record = await result.single()
            if record is not None:
                return list(record["remaining"] or [])

        return []

    async def get_document_references(
        self,
        id: str,
        namespace: str,
    ) -> List[str]:
        """Return current ``source_doc_ids`` for a node or edge."""
        query_node = """
        MATCH (n:Entity {id: $id})
        WHERE $namespace IN n.namespaces
        RETURN n.source_doc_ids AS doc_ids
        """
        async with self.driver.session() as session:
            result = await session.run(query_node, id=id, namespace=namespace)
            record = await result.single()
            if record is not None:
                return list(record["doc_ids"] or [])

        query_edge = """
        MATCH ()-[r]->()
        WHERE r.id = $id AND $namespace IN r.namespaces
        RETURN r.source_doc_ids AS doc_ids
        """
        async with self.driver.session() as session:
            result = await session.run(query_edge, id=id, namespace=namespace)
            record = await result.single()
            if record is not None:
                return list(record["doc_ids"] or [])

        return []

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_neighbors(
        self,
        node_id: str,
        namespace: str,
        direction: str = "both",
        edge_types: Optional[List[str]] = None,
    ) -> List[GraphNode]:
        if direction == "out":
            pattern = "(n:Entity {id: $id})-[r]->(m:Entity)"
        elif direction == "in":
            pattern = "(m:Entity)-[r]->(n:Entity {id: $id})"
        else:
            pattern = "(n:Entity {id: $id})-[r]-(m:Entity)"

        where_parts = ["$namespace IN r.namespaces", "$namespace IN m.namespaces"]
        params: Dict[str, Any] = {"id": node_id, "namespace": namespace}

        if edge_types:
            where_parts.append("type(r) IN $edge_types")
            params["edge_types"] = edge_types

        where_str = " AND ".join(where_parts)

        query = f"""
        MATCH {pattern}
        WHERE {where_str}
        RETURN DISTINCT m
        """
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._node_from_record(dict(r["m"]), namespace) for r in records]

    async def query_nodes(
        self,
        filters: Dict[str, Any],
        namespace: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where_clauses = []
        params: Dict[str, Any] = {"limit": limit}

        if namespace:
            where_clauses.append("$namespace IN n.namespaces")
            params["namespace"] = namespace

        for idx, (k, v) in enumerate(filters.items()):
            param_name = f"p{idx}"
            where_clauses.append(f"n.{k} = ${param_name}")
            params[param_name] = v

        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (n:Entity)
        {where_str}
        RETURN n
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [dict(r["n"]) for r in records]

    async def get_subgraph(
        self,
        node_ids: List[str],
        namespace: str,
        max_hops: int = 2,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        if not node_ids:
            return [], []

        query = """
        MATCH (n:Entity)
        WHERE n.id IN $seed_ids AND $namespace IN n.namespaces
        WITH collect(n) AS seeds
        CALL apoc.path.subgraphNodes(seeds, {maxLevel: $max_hops}) YIELD node
        WITH collect(DISTINCT node) AS nodes
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE s IN nodes AND t IN nodes AND $namespace IN r.namespaces
        RETURN nodes, collect(DISTINCT r) AS rels
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                seed_ids=node_ids,
                namespace=namespace,
                max_hops=max_hops,
            )
            record = await result.single()
            if not record:
                return [], []

            nodes = []
            for n in record["nodes"]:
                nodes.append(self._node_from_record(dict(n), namespace))

            edges = []
            for r in record["rels"]:
                edges.append(self._edge_from_record(r, namespace))

            return nodes, edges

    # ------------------------------------------------------------------
    # Personalized PageRank
    # ------------------------------------------------------------------

    async def personalized_pagerank(
        self,
        seed_nodes: List[str],
        namespace: str,
        damping: float = 0.85,
        max_iterations: int = 100,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        query = """
        CALL gds.pageRank.stream(
          {
            nodeQuery: 'MATCH (n:Entity) WHERE $namespace IN n.namespaces RETURN id(n) as id',
            relationshipQuery: 'MATCH (n:Entity)-[r]->(m:Entity) WHERE $namespace IN r.namespaces RETURN id(n) as source, id(m) as target, r.weight as weight',
            relationshipWeightProperty: 'weight',
            dampingFactor: $damping,
            maxIterations: $iterations,
            sourceNodes: $source_nodes
          }
        )
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        async with self.driver.session() as session:
            id_query = """
            MATCH (n:Entity)
            WHERE n.id IN $seeds AND $namespace IN n.namespaces
            RETURN id(n) AS internal_id
            """
            result = await session.run(
                id_query, seeds=seed_nodes, namespace=namespace
            )
            internal_ids = [record["internal_id"] async for record in result]

            if not internal_ids:
                return []

            result = await session.run(
                query,
                namespace=namespace,
                damping=damping,
                iterations=max_iterations,
                source_nodes=internal_ids,
                top_k=top_k,
            )

            ppr_results = []
            async for record in result:
                ppr_results.append((record["id"], record["score"]))

            return ppr_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_from_record(props: Dict[str, Any], namespace: str) -> GraphNode:
        """Build a GraphNode from a Neo4j record dict."""
        nid = props.pop("id", "")
        node_type_str = props.pop("node_type", "entity")
        content = props.pop("content", "")
        props.pop("namespaces", None)

        src_doc_ids = props.pop("source_doc_ids", [])
        src_chunk_indices = props.pop("source_chunk_indices", [])
        source_locations = parallel_arrays_to_source_locations(
            src_doc_ids, src_chunk_indices
        )

        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.ENTITY

        return EntityNode(
            id=nid,
            node_type=node_type,
            content=content,
            source_locations=source_locations,
            namespace=namespace,
            properties=props,
            entity_name=props.pop("label", nid),
        )

    @staticmethod
    def _edge_from_record(rel: Any, namespace: str) -> GraphEdge:
        """Build a GraphEdge from a Neo4j relationship record."""
        props = dict(rel)
        start_id = rel.start_node["id"]
        end_id = rel.end_node["id"]
        rid = props.pop("id", None)
        weight = props.pop("weight", 1.0)
        is_bidir = props.pop("is_bidirectional", False)
        inv_rel = props.pop("inverse_relation", None)
        props.pop("namespaces", None)

        src_doc_ids = props.pop("source_doc_ids", [])
        src_chunk_indices = props.pop("source_chunk_indices", [])
        source_locations = parallel_arrays_to_source_locations(
            src_doc_ids, src_chunk_indices
        )

        return GraphEdge(
            source_id=start_id,
            target_id=end_id,
            relation=rel.type,
            id=rid,
            weight=weight,
            is_bidirectional=is_bidir,
            inverse_relation=inv_rel,
            source_locations=source_locations,
            namespace=namespace,
            properties=props,
        )
