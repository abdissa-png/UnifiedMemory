"""
In-memory implementation of GraphStoreBackend using NetworkX.
Useful for testing and local development.
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
    Namespace isolation is handled by checking 'namespace' attribute on nodes/edges.
    """

    def __init__(self) -> None:
        # We use a single graph for all namespaces, similar to how a DB might work,
        # but filter on access.
        self._graph = nx.MultiDiGraph()
        self._lock = asyncio.Lock()

    async def create_node(
        self,
        node: GraphNode,
        namespace: str,
    ) -> str:
        """Create a node."""
        async with self._lock:
            # Store all attributes
            attrs = {
                "id": node.id,
                "node_type": node.node_type.value,
                "content": node.content,
                "namespace": namespace,
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
            attrs = {
                "id": edge_id,
                "relation": edge.relation,
                "weight": edge.weight,
                "namespace": namespace,
                **edge.properties
            }
            
            self._graph.add_edge(edge.source_id, edge.target_id, key=edge_id, **attrs)
            
            if edge.is_bidirectional:
                # Add inverse edge
                inverse_attrs = attrs.copy()
                inverse_attrs["id"] = f"{edge_id}_inv"
                inverse_attrs["relation"] = edge.inverse_relation or edge.relation
                self._graph.add_edge(edge.target_id, edge.source_id, key=inverse_attrs["id"], **inverse_attrs)
                
            return edge_id

    async def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        namespace: str,
    ) -> List[str]:
        """Batch create nodes."""
        async with self._lock:
            ids = []
            for node in nodes:
                attrs = {
                    "id": node.id,
                    "node_type": node.node_type.value,
                    "content": node.content,
                    "namespace": namespace,
                    **node.properties
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
        """Batch create edges."""
        async with self._lock:
            ids = []
            for edge in edges:
                edge_id = edge.id or str(uuid.uuid4())
                attrs = {
                    "id": edge_id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "namespace": namespace,
                    **edge.properties
                }
                self._graph.add_edge(edge.source_id, edge.target_id, key=edge_id, **attrs)
                ids.append(edge_id)
                
                if edge.is_bidirectional:
                    inverse_attrs = attrs.copy()
                    inverse_attrs["id"] = f"{edge_id}_inv"
                    inverse_attrs["relation"] = edge.inverse_relation or edge.relation
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
            if data.get("namespace") != namespace:
                return None
                
            # Reconstruct node object (simplified generic reconstruction)
            # In a real app we'd map back to proper EntityNode/ChunkNode types
            return self._dict_to_node(data)

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
                    if data.get("namespace") == namespace:
                        results.append(self._dict_to_node(data))
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
            if self._graph.nodes[node_id].get("namespace") != namespace:
                return []
                
            neighbors_ids = set()
            
            if direction in ("out", "both"):
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    if data.get("namespace") == namespace:
                        if edge_types and data.get("relation") not in edge_types:
                            continue
                        neighbors_ids.add(target)
                        
            if direction in ("in", "both"):
                for source, _, data in self._graph.in_edges(node_id, data=True):
                    if data.get("namespace") == namespace:
                        if edge_types and data.get("relation") not in edge_types:
                            continue
                        neighbors_ids.add(source)
            
            results = []
            for nid in neighbors_ids:
                if self._graph.has_node(nid):
                    node_data = self._graph.nodes[nid]
                    if node_data.get("namespace") == namespace:
                         results.append(self._dict_to_node(node_data))
            
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
                if namespace and data.get("namespace") != namespace:
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
        """Delete node."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False
                
            data = self._graph.nodes[node_id]
            if data.get("namespace") != namespace:
                return False
                
            self._graph.remove_node(node_id)
            return True

    async def delete_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete edges."""
        async with self._lock:
            to_remove = []
            
            # Inefficient scan for NetworkX (which is optimized for adjacency, not edge list)
            # But acceptable for testing
            for u, v, k, data in self._graph.edges(keys=True, data=True):
                if namespace and data.get("namespace") != namespace:
                    continue
                if source_id and u != source_id:
                    continue
                if target_id and v != target_id:
                    continue
                
                to_remove.append((u, v, k))
                
            for u, v, k in to_remove:
                self._graph.remove_edge(u, v, key=k)
                
            return len(to_remove)

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
            # Filter nodes by namespace
            ns_nodes = [n for n, d in self._graph.nodes(data=True) if d.get("namespace") == namespace]
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
                    if self._graph.nodes[nid].get("namespace") != namespace:
                        continue
                        
                    # Out neighbors
                    for _, target, data in self._graph.out_edges(nid, data=True):
                        if data.get("namespace") == namespace:
                            if target not in visited:
                                visited.add(target)
                                next_layer.add(target)
                                
                    # In neighbors
                    for source, _, data in self._graph.in_edges(nid, data=True):
                        if data.get("namespace") == namespace:
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
                    if data.get("namespace") == namespace:
                        nodes.append(self._dict_to_node(data))
            
            # Collect Edges between visited nodes
            edges = []
            subgraph = self._graph.subgraph(list(visited))
            for u, v, k, data in subgraph.edges(keys=True, data=True):
                 if data.get("namespace") == namespace:
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

    def _dict_to_node(self, data: Dict[str, Any]) -> GraphNode:
        """Helper to convert stored dict back to GraphNode."""
        from unified_memory.core.types import EntityNode, PassageNode, PageNode, NodeType
        
        node_type_str = data.get("node_type", "entity")
        
        # Base args
        base_args = {
            "id": data["id"],
            "node_type": NodeType(node_type_str),
            "content": data.get("content", ""),
            "namespace": data.get("namespace", "default"),
            "properties": {k: v for k, v in data.items() if k not in ["id", "node_type", "content", "namespace", "entity_name", "entity_type", "incoming_edge_count", "outgoing_edge_count"]}
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
