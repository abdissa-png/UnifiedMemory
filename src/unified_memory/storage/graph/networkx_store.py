"""
In-memory implementation of GraphStoreBackend using NetworkX.
Useful for testing and local development.

NAMESPACE ARRAY MIGRATION:
- Nodes and edges now store `namespaces: List[str]` instead of `namespace: str`.
- Backward compat: reads both `namespaces` (new) and `namespace` (old).
- Filtering matches if query namespace is IN the namespaces list.
"""

from __future__ import annotations

import asyncio
import uuid
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Set

from unified_memory.storage.base import GraphStoreBackend
from unified_memory.core.types import GraphNode, GraphEdge, NodeType


class NetworkXGraphStore(GraphStoreBackend):
    """
    In-memory Graph store implementation using NetworkX.
    
    Uses a MultiDiGraph to support directed edges with multiple relations between nodes.
    Namespace isolation is handled by checking 'namespaces' attribute on nodes/edges.
    """

    def __init__(self) -> None:
        # We use a single graph for all namespaces, similar to how a DB might work,
        # but filter on access.
        self._graph = nx.MultiDiGraph()
        self._lock = asyncio.Lock()

    def _get_namespaces(self, data: Dict[str, Any]) -> List[str]:
        """Extract namespaces list from node/edge data (backward compat)."""
        if "namespaces" in data:
            return data["namespaces"]
        if "namespace" in data:
            return [data["namespace"]]
        return []

    def _ns_matches(self, data: Dict[str, Any], namespace: str) -> bool:
        """Check if namespace is in the stored namespaces list."""
        return namespace in self._get_namespaces(data)

    async def create_node(
        self,
        node: GraphNode,
        namespace: str,
    ) -> str:
        """Create a node."""
        async with self._lock:
            # If node already exists, merge namespaces and provenance
            if self._graph.has_node(node.id):
                existing_data = self._graph.nodes[node.id]
                existing_ns = self._get_namespaces(existing_data)
                if namespace not in existing_ns:
                    existing_ns.append(namespace)
                    self._graph.nodes[node.id]["namespaces"] = existing_ns
                
                # Merge source_locations
                new_locs = getattr(node, "source_locations", [])
                if new_locs:
                    existing_doc_ids = existing_data.get("source_doc_ids", [])
                    existing_chunk_indices = existing_data.get("source_chunk_indices", [])
                    existing_pairs = set(zip(existing_doc_ids, existing_chunk_indices))
                    for loc in new_locs:
                        pair = (loc.document_id, loc.chunk_index)
                        if pair not in existing_pairs:
                            existing_doc_ids.append(loc.document_id)
                            existing_chunk_indices.append(loc.chunk_index)
                            existing_pairs.add(pair)
                    self._graph.nodes[node.id]["source_doc_ids"] = existing_doc_ids
                    self._graph.nodes[node.id]["source_chunk_indices"] = existing_chunk_indices
                
                return node.id
            
            from unified_memory.core.types import source_locations_to_parallel_arrays
            provenance = source_locations_to_parallel_arrays(
                getattr(node, "source_locations", [])
            )
            # Store all attributes
            attrs = {
                "id": node.id,
                "node_type": node.node_type.value,
                "content": node.content,
                "namespaces": [namespace],
                "source_doc_ids": provenance["source_doc_ids"],
                "source_chunk_indices": provenance["source_chunk_indices"],
                **node.properties
            }
            # Special handling for specific node types
            if hasattr(node, "entity_name"):
                 attrs["entity_name"] = node.entity_name
            if hasattr(node, "entity_type"):
                 attrs["entity_type"] = node.entity_type
                 
            self._graph.add_node(node.id, **attrs)
            return node.id

    async def create_edge(
        self,
        edge: GraphEdge,
        namespace: str,
    ) -> str:
        """Create an edge."""
        async with self._lock:
            edge_id = edge.id or str(uuid.uuid4())
            
            # Check if edge already exists? nx.MultiDiGraph.has_edge doesn't check key easily
            # But we use edge_id as key.
            # For simplicity, if edge_id is provided, check if it exists in any edge data
            from unified_memory.core.types import source_locations_to_parallel_arrays
            provenance = source_locations_to_parallel_arrays(
                getattr(edge, "source_locations", [])
            )
            
            attrs = {
                "id": edge_id,
                "relation": edge.relation,
                "weight": edge.weight,
                "namespaces": [namespace],
                "source_doc_ids": provenance["source_doc_ids"],
                "source_chunk_indices": provenance["source_chunk_indices"],
                **edge.properties
            }
            
            self._graph.add_edge(edge.source_id, edge.target_id, key=edge_id, **attrs)
            
            if edge.is_bidirectional:
                # Add inverse edge
                inverse_attrs = attrs.copy()
                inverse_attrs["id"] = f"{edge_id}_inv"
                inverse_attrs["relation"] = edge.inverse_relation or edge.relation
                inverse_attrs["namespaces"] = [namespace]
                self._graph.add_edge(edge.target_id, edge.source_id, key=inverse_attrs["id"], **inverse_attrs)
                
            return edge_id

    async def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        namespace: str,
    ) -> List[str]:
        """Batch create nodes with dedup and source provenance merging."""
        async with self._lock:
            ids = []
            for node in nodes:
                if self._graph.has_node(node.id):
                    existing_data = self._graph.nodes[node.id]
                    existing_ns = self._get_namespaces(existing_data)
                    if namespace not in existing_ns:
                        existing_ns.append(namespace)
                        self._graph.nodes[node.id]["namespaces"] = existing_ns

                    # Merge source_locations (parallel arrays)
                    new_locs = getattr(node, "source_locations", [])
                    if new_locs:
                        existing_doc_ids = existing_data.get("source_doc_ids", [])
                        existing_chunk_indices = existing_data.get("source_chunk_indices", [])
                        existing_pairs = set(zip(existing_doc_ids, existing_chunk_indices))
                        for loc in new_locs:
                            pair = (loc.document_id, loc.chunk_index)
                            if pair not in existing_pairs:
                                existing_doc_ids.append(loc.document_id)
                                existing_chunk_indices.append(loc.chunk_index)
                                existing_pairs.add(pair)
                        self._graph.nodes[node.id]["source_doc_ids"] = existing_doc_ids
                        self._graph.nodes[node.id]["source_chunk_indices"] = existing_chunk_indices

                    ids.append(node.id)
                    continue

                from unified_memory.core.types import source_locations_to_parallel_arrays
                provenance = source_locations_to_parallel_arrays(
                    getattr(node, "source_locations", [])
                )
                attrs = {
                    "id": node.id,
                    "node_type": node.node_type.value,
                    "content": node.content,
                    "namespaces": [namespace],
                    "source_doc_ids": provenance["source_doc_ids"],
                    "source_chunk_indices": provenance["source_chunk_indices"],
                    **node.properties,
                }
                if hasattr(node, "entity_name"):
                    attrs["entity_name"] = node.entity_name
                if hasattr(node, "entity_type"):
                    attrs["entity_type"] = node.entity_type

                self._graph.add_node(node.id, **attrs)
                ids.append(node.id)
            return ids

    async def create_edges_batch(
        self,
        edges: List[GraphEdge],
        namespace: str,
    ) -> List[str]:
        """Batch create edges with source provenance."""
        async with self._lock:
            ids = []
            for edge in edges:
                edge_id = edge.id or str(uuid.uuid4())
                from unified_memory.core.types import source_locations_to_parallel_arrays
                provenance = source_locations_to_parallel_arrays(
                    getattr(edge, "source_locations", [])
                )
                attrs = {
                    "id": edge_id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "namespaces": [namespace],
                    "source_doc_ids": provenance["source_doc_ids"],
                    "source_chunk_indices": provenance["source_chunk_indices"],
                    **edge.properties,
                }
                self._graph.add_edge(edge.source_id, edge.target_id, key=edge_id, **attrs)
                ids.append(edge_id)

                if edge.is_bidirectional:
                    inverse_attrs = attrs.copy()
                    inverse_attrs["id"] = f"{edge_id}_inv"
                    inverse_attrs["relation"] = edge.inverse_relation or edge.relation
                    inverse_attrs["namespaces"] = [namespace]
                    self._graph.add_edge(edge.target_id, edge.source_id, key=inverse_attrs["id"], **inverse_attrs)
                    
            return ids

    async def get_node(
        self,
        node_id: str,
        namespace: str,
    ) -> Optional[GraphNode]:
        """Get node by ID."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return None
            
            data = self._graph.nodes[node_id]
            if not self._ns_matches(data, namespace):
                return None
                
            return self._dict_to_node(data, namespace)

    async def get_nodes_batch(
        self,
        node_ids: List[str],
        namespace: str,
    ) -> List[GraphNode]:
        """Batch get nodes."""
        async with self._lock:
            results = []
            for nid in node_ids:
                if self._graph.has_node(nid):
                    data = self._graph.nodes[nid]
                    if self._ns_matches(data, namespace):
                        results.append(self._dict_to_node(data, namespace))
            return results

    async def get_neighbors(
        self,
        node_id: str,
        namespace: str,
        direction: str = "both",
        edge_types: Optional[List[str]] = None,
    ) -> List[GraphNode]:
        """Get neighboring nodes."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return []
            
            # Verify source node access
            if not self._ns_matches(self._graph.nodes[node_id], namespace):
                return []
                
            neighbors_ids = set()
            
            if direction in ("out", "both"):
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    if self._ns_matches(data, namespace):
                        if edge_types and data.get("relation") not in edge_types:
                            continue
                        neighbors_ids.add(target)
                        
            if direction in ("in", "both"):
                for source, _, data in self._graph.in_edges(node_id, data=True):
                    if self._ns_matches(data, namespace):
                        if edge_types and data.get("relation") not in edge_types:
                            continue
                        neighbors_ids.add(source)
            
            results = []
            for nid in neighbors_ids:
                if self._graph.has_node(nid):
                    node_data = self._graph.nodes[nid]
                    if self._ns_matches(node_data, namespace):
                         results.append(self._dict_to_node(node_data, namespace))
            
            return results

    async def query_nodes(
        self,
        filters: Dict[str, Any],
        namespace: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query nodes by filter."""
        async with self._lock:
            results = []
            for nid, data in self._graph.nodes(data=True):
                if namespace and not self._ns_matches(data, namespace):
                    continue
                
                match = True
                for k, v in filters.items():
                    if data.get(k) != v:
                        match = False
                        break
                
                if match:
                    results.append(data.copy())
                    if len(results) >= limit:
                        break
            return results

    async def delete_node(
        self,
        node_id: str,
        namespace: str,
    ) -> bool:
        """
        Delete node for a specific namespace.

        Ref-count semantics:
        - Remove the namespace from the node's namespaces list.
        - Hard-delete the node only when no namespaces remain.
        """
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False

            data = self._graph.nodes[node_id]
            ns_list = self._get_namespaces(data)
            if namespace not in ns_list:
                return False

            ns_list.remove(namespace)
            if not ns_list:
                self._graph.remove_node(node_id)
                return True

            self._graph.nodes[node_id]["namespaces"] = ns_list
            return True

    async def delete_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete edges for a specific namespace.

        Ref-count semantics:
        - For each matching edge, remove the namespace from its namespaces list.
        - Hard-delete the edge only when no namespaces remain.
        """
        async with self._lock:
            to_remove = []

            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if namespace and not self._ns_matches(data, namespace):
                    continue
                if source_id and u != source_id:
                    continue
                if target_id and v != target_id:
                    continue

                to_remove.append((u, v, k))

            removed_count = 0
            for u, v, k in to_remove:
                data = self._graph.edges[u, v, k]
                ns_list = self._get_namespaces(data)
                if namespace and namespace in ns_list:
                    ns_list.remove(namespace)
                    if not ns_list:
                        self._graph.remove_edge(u, v, key=k)
                    else:
                        self._graph.edges[u, v, k]["namespaces"] = ns_list
                    removed_count += 1

            return removed_count

    async def add_namespace_to_node(
        self,
        node_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing node."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False
            data = self._graph.nodes[node_id]
            ns_list = self._get_namespaces(data)
            if namespace not in ns_list:
                ns_list.append(namespace)
                self._graph.nodes[node_id]["namespaces"] = ns_list
            
            if document_id:
                doc_ids = data.get("source_doc_ids", [])
                if document_id not in doc_ids:
                    doc_ids.append(document_id)
                    self._graph.nodes[node_id]["source_doc_ids"] = doc_ids
            return True

    async def add_namespace_to_edge(
        self,
        edge_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing edge."""
        async with self._lock:
            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if k == edge_id:
                    ns_list = self._get_namespaces(data)
                    if namespace not in ns_list:
                        ns_list.append(namespace)
                        self._graph.edges[u, v, k]["namespaces"] = ns_list
                    
                    if document_id:
                        doc_ids = data.get("source_doc_ids", [])
                        if document_id not in doc_ids:
                            doc_ids.append(document_id)
                            self._graph.edges[u, v, k]["source_doc_ids"] = doc_ids
                    return True
            return False

    async def add_namespace(
        self,
        id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add namespace to a node or edge (tries node first, then edge)."""
        if await self.add_namespace_to_node(id, namespace, document_id):
            return True
        return await self.add_namespace_to_edge(id, namespace, document_id)

    async def remove_namespace_from_node(
        self,
        node_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove namespace from a node. Returns (success, was_last)."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False, False
            data = self._graph.nodes[node_id]
            ns_list = self._get_namespaces(data)
            if namespace not in ns_list:
                return False, False
            ns_list.remove(namespace)
            if not ns_list:
                self._graph.remove_node(node_id)
                return True, True
            self._graph.nodes[node_id]["namespaces"] = ns_list
            return True, False

    async def remove_namespace_from_edge(
        self,
        edge_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove namespace from an edge. Returns (success, was_last)."""
        async with self._lock:
            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if k == edge_id:
                    ns_list = self._get_namespaces(data)
                    if namespace not in ns_list:
                        return False, False
                    ns_list.remove(namespace)
                    if not ns_list:
                        self._graph.remove_edge(u, v, key=k)
                        return True, True
                    self._graph.edges[u, v, k]["namespaces"] = ns_list
                    return True, False
            return False, False

    async def remove_namespace(
        self,
        id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove namespace from a node or edge (tries node first, then edge)."""
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
        ``source_chunk_indices`` on a node or edge without touching
        namespaces.  Returns remaining ``source_doc_ids``."""
        async with self._lock:
            # Try node first
            if self._graph.has_node(id):
                data = self._graph.nodes[id]
                doc_ids = list(data.get("source_doc_ids") or [])
                chunk_idxs = list(data.get("source_chunk_indices") or [])
                new_doc_ids = []
                new_chunk_idxs = []
                for d, c in zip(doc_ids, chunk_idxs):
                    if d != document_id:
                        new_doc_ids.append(d)
                        new_chunk_idxs.append(c)
                self._graph.nodes[id]["source_doc_ids"] = new_doc_ids
                self._graph.nodes[id]["source_chunk_indices"] = new_chunk_idxs
                return new_doc_ids

            # Try edge
            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if k == id:
                    doc_ids = list(data.get("source_doc_ids") or [])
                    chunk_idxs = list(data.get("source_chunk_indices") or [])
                    new_doc_ids = []
                    new_chunk_idxs = []
                    for d, c in zip(doc_ids, chunk_idxs):
                        if d != document_id:
                            new_doc_ids.append(d)
                            new_chunk_idxs.append(c)
                    self._graph.edges[u, v, k]["source_doc_ids"] = new_doc_ids
                    self._graph.edges[u, v, k]["source_chunk_indices"] = new_chunk_idxs
                    return new_doc_ids

        return []

    async def get_document_references(
        self,
        id: str,
        namespace: str,
    ) -> List[str]:
        """Return current ``source_doc_ids`` for a node or edge."""
        async with self._lock:
            if self._graph.has_node(id):
                data = self._graph.nodes[id]
                if self._ns_matches(data, namespace):
                    return list(data.get("source_doc_ids") or [])

            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if k == id and self._ns_matches(data, namespace):
                    return list(data.get("source_doc_ids") or [])

        return []

    async def personalized_pagerank(
        self,
        seed_nodes: List[str],
        namespace: str,
        damping: float = 0.85,
        max_iterations: int = 100,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Run Personalized PageRank."""
        async with self._lock:
            # Create subgraph for namespace
            ns_nodes = [n for n, d in self._graph.nodes(data=True) if self._ns_matches(d, namespace)]
            subgraph = self._graph.subgraph(ns_nodes)
            
            if not seed_nodes:
                return []
                
            # Filter seeds that exist in subgraph
            valid_seeds = {s: 1.0 for s in seed_nodes if subgraph.has_node(s)}
            if not valid_seeds:
                return []
            
            try:
                scores = nx.pagerank(
                    subgraph,
                    alpha=damping,
                    personalization=valid_seeds,
                    max_iter=max_iterations
                )
            except nx.PowerIterationFailedConvergence:
                return []
                
            # Sort and return top_k
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_scores[:top_k]

    async def get_subgraph(
        self,
        node_ids: List[str],
        namespace: str,
        max_hops: int = 2,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get subgraph around nodes."""
        async with self._lock:
            # BFS from seeds
            visited = set(node_ids)
            current_layer = set(node_ids)
            
            for _ in range(max_hops):
                next_layer = set()
                for nid in current_layer:
                    if not self._graph.has_node(nid):
                        continue
                    if not self._ns_matches(self._graph.nodes[nid], namespace):
                        continue
                        
                    # Out neighbors
                    for _, target, data in self._graph.out_edges(nid, data=True):
                        if self._ns_matches(data, namespace):
                            if target not in visited:
                                visited.add(target)
                                next_layer.add(target)
                                
                    # In neighbors
                    for source, _, data in self._graph.in_edges(nid, data=True):
                        if self._ns_matches(data, namespace):
                            if source not in visited:
                                visited.add(source)
                                next_layer.add(source)
                current_layer = next_layer
                if not current_layer:
                    break
            
            # Collect Nodes
            nodes = []
            for nid in visited:
                 if self._graph.has_node(nid):
                    data = self._graph.nodes[nid]
                    if self._ns_matches(data, namespace):
                        nodes.append(self._dict_to_node(data, namespace))
            
            # Collect Edges between visited nodes
            edges = []
            subgraph = self._graph.subgraph(list(visited))
            for u, v, k, data in subgraph.edges(keys=True, data=True):
                 if self._ns_matches(data, namespace):
                     edges.append(GraphEdge(
                         id=data.get("id", k),
                         source_id=u,
                         target_id=v,
                         relation=data.get("relation", "related"),
                         weight=data.get("weight", 1.0),
                         namespace=namespace,
                         properties=data
                     ))
                     
            return nodes, edges

    def _dict_to_node(self, data: Dict[str, Any], namespace: str = "default") -> GraphNode:
        """Helper to convert stored dict back to GraphNode."""
        from unified_memory.core.types import EntityNode, PassageNode, PageNode, NodeType
        
        node_type_str = data.get("node_type", "entity")
        
        # Use the queried namespace for the reconstructed node
        node_namespace = namespace
        
        # Base args — exclude internal fields from properties
        excluded_keys = {"id", "node_type", "content", "namespace", "namespaces", 
                         "entity_name", "entity_type", "incoming_edge_count", "outgoing_edge_count"}
        base_args = {
            "id": data["id"],
            "node_type": NodeType(node_type_str),
            "content": data.get("content", ""),
            "namespace": node_namespace,
            "properties": {k: v for k, v in data.items() if k not in excluded_keys}
        }

        if node_type_str == "entity":
            return EntityNode(
                **base_args,
                entity_name=data.get("entity_name", data.get("name", "")),
                entity_type=data.get("entity_type", "entity"),
                incoming_edge_count=data.get("incoming_edge_count", 0),
                outgoing_edge_count=data.get("outgoing_edge_count", 0)
            )
        elif node_type_str == "passage":
            # Just minimal reconstruction for test purposes
            return PassageNode(
                 **base_args
            )
        elif node_type_str == "page":
             return PageNode(
                 **base_args
             )
        
        # Fallback to base node if we could instantiate it, but it's abstract-ish because implementations expect subclasses
        # For now return EntityNode as default container
        return EntityNode(**base_args)
