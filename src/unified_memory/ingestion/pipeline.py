"""
Ingestion Pipeline Orchestrator.

Coordinates the flow: Document -> Parse -> Chunk -> Embed -> Store (CAS, Vector, Graph).
Implements the Saga pattern for atomic-like processing of documents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from unified_memory.core.types import Chunk, SourceReference, SourceType, Modality
from unified_memory.ingestion.parsers.base import DocumentParser, ParsedDocument
from unified_memory.ingestion.chunkers.base import Chunker
from unified_memory.embeddings.base import EmbeddingProvider
from unified_memory.storage.base import VectorStoreBackend, GraphStoreBackend
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.ingestion.extractors.base import Extractor
logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a document."""
    document_id: str
    source: SourceReference
    chunk_count: int = 0
    page_count: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class ParserRegistry:
    """Registry of available document parsers."""
    def __init__(self) -> None:
        self._ext_map: Dict[str, DocumentParser] = {}
        self._mime_map: Dict[str, DocumentParser] = {}
    
    def register(self, parser: DocumentParser) -> None:
        for ext in parser.supported_extensions:
            self._ext_map[ext.lower()] = parser
        for mime in parser.supported_mime_types:
            self._mime_map[mime.lower()] = parser
    
    def get_parser_for_file(self, path: Path) -> Optional[DocumentParser]:
        return self._ext_map.get(path.suffix.lower())


class IngestionPipeline:
    """
    Enhanced ingestion pipeline with CAS, Embedding, and Saga support.
    
    Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md and INITIAL_PLAN.md
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreBackend,
        cas_registry: CASRegistry,
        content_store: ContentStore,
        graph_store: Optional[GraphStoreBackend] = None,
        chunker: Optional[Chunker] = None,
        parser_registry: Optional[ParserRegistry] = None,
    ) -> None:
        from unified_memory.ingestion.parsers.text import TextParser
        from unified_memory.ingestion.chunkers.fixed_size import FixedSizeChunker
        
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.cas_registry = cas_registry
        self.content_store = content_store
        self.graph_store = graph_store
        self.extractors: List[Extractor] = []
        
        self.chunker = chunker or FixedSizeChunker()
        self.parser_registry = parser_registry or ParserRegistry()
        
        if parser_registry is None:
            self.parser_registry.register(TextParser())

    async def _process_chunks(
        self, 
        chunks: List[Chunk], 
        namespace: str,
        embedding_model: str,
        parsed_document: Optional[ParsedDocument] = None,
    ) -> List[str]:
        """
        Process a batch of chunks through Embed -> CAS -> Storage.
        
        Returns a list of error messages, if any.
        """
        if not chunks:
            return []

        # 1. Batch Embedding
        try:
            contents = [c.content for c in chunks]
            embeddings = await self.embedding_provider.embed_batch(
                contents, modality=Modality.TEXT
            )
            for i, emb in enumerate(embeddings):
                chunks[i].embedding = emb
        except Exception as e:
            logger.exception("Embedding failed")
            return [f"Embedding error: {str(e)}"]

        # 1.5 Structural Graph Storage (Page Nodes)
        if self.graph_store and parsed_document:
            from unified_memory.core.types import PageNode, NodeType
            for page in parsed_document.pages:
                page_node = PageNode(
                    id=f"page:{parsed_document.document_id}:{page.page_number}",
                    content=page.full_text[:200], # snippet
                    namespace=namespace,
                    page_number=page.page_number,
                    document_id=parsed_document.document_id,
                    node_type=NodeType.PAGE
                )
                try:
                    await self.graph_store.create_node(page_node, namespace)
                except Exception as e:
                    logger.warning(f"Failed to create page node: {e}")

        # 2. Storage with Saga-like cleanup (partial implementation for now)
        # In a real Saga, we'd track exactly what was created to roll back.
        # Here we prioritize atomicity per chunk.
        
        errors = []
        for i, chunk in enumerate(chunks):
            try:
                # Content ID is often same as content hash in CAS
                content_id = chunk.content_hash
                
                # CAS Registration
                await self.cas_registry.register(
                    content_hash=chunk.content_hash,
                    content_id=content_id,
                )
                
                # Add reference
                await self.cas_registry.add_reference(
                    content_hash=chunk.content_hash,
                    namespace=namespace,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index
                )
                
                # Content Storage (deduplicated)
                await self.content_store.store_content(content_id, chunk.content)
                
            # Vector Storage
                vector_data = {
                    "id": f"{chunk.document_id}:{chunk.chunk_index}",
                    "embedding": chunk.embedding,
                    "metadata": {
                        **chunk.metadata,
                        "content": chunk.content, # duplicate for convenience in search
                        "content_hash": chunk.content_hash,
                    }
                }
                await self.vector_store.upsert([vector_data], namespace)
                
                # 3. Extraction and Graph Storage
                if self.graph_store:
                    from unified_memory.core.types import EntityNode, GraphEdge, NodeType, PassageNode
                    
                    # Store PassageNode (the chunk itself in the graph)
                    passage_node = PassageNode(
                        id=f"chunk:{chunk.document_id}:{chunk.chunk_index}",
                        content=chunk.content,
                        namespace=namespace,
                        node_type=NodeType.PASSAGE,
                        properties={**chunk.metadata, "content_hash": chunk.content_hash}
                    )
                    await self.graph_store.create_node(passage_node, namespace)
                    
                    # Link to Page node if available
                    if chunk.page_number is not None:
                        link_edge = GraphEdge(
                            source_id=f"page:{chunk.document_id}:{chunk.page_number}",
                            target_id=passage_node.id,
                            relation="HAS_CHUNK",
                            namespace=namespace
                        )
                        await self.graph_store.create_edge(link_edge, namespace)

                    if self.extractors:
                        for extractor in self.extractors:
                            extraction = await extractor.extract(chunk)
                            
                            # Store entities
                            for ent in extraction.entities:
                                node = EntityNode(
                                    id=ent.name,
                                    content=ent.description or "",
                                    namespace=namespace,
                                    entity_name=ent.name,
                                    entity_type=ent.type,
                                    properties=ent.properties,
                                    node_type=NodeType.ENTITY
                                )
                                await self.graph_store.create_node(node, namespace)
                                
                                # Link chunk to entity
                                ent_edge = GraphEdge(
                                    source_id=passage_node.id,
                                    target_id=node.id,
                                    relation="MENTIONS",
                                    namespace=namespace
                                )
                                await self.graph_store.create_edge(ent_edge, namespace)
                                
                            # Store relations
                            for rel in extraction.relations:
                                edge = GraphEdge(
                                    source_id=rel.source_entity,
                                    target_id=rel.target_entity,
                                    relation=rel.relation_type,
                                    properties=rel.properties,
                                    namespace=namespace
                                )
                                await self.graph_store.create_edge(edge, namespace)
                
            except Exception as e:
                logger.exception(f"Failed to process chunk {i}")
                errors.append(f"Chunk {i} processing error: {str(e)}")
                # TODO: Implement rollback logic for standard Saga pattern (P1 issue #5)
        
        return errors

    async def ingest_file(
        self,
        path: Path,
        namespace: str = "default",
        document_id: Optional[str] = None,
        **options: Any,
    ) -> IngestionResult:
        doc_id = document_id or str(uuid.uuid4())
        source_ref = SourceReference(source_id=doc_id, source_type=SourceType.TEXT_BLOCK)
        
        parser = self.parser_registry.get_parser_for_file(path)
        if not parser:
            return IngestionResult(doc_id, source_ref, errors=[f"No parser for {path.suffix}"])
        
        try:
            parsed = await parser.parse_file(path, doc_id, **options)
            
            embedding_model = options.get("embedding_model", self.embedding_provider.model_id)
            chunks = await self.chunker.chunk(parsed, namespace, embedding_model=embedding_model)
            
            processing_errors = await self._process_chunks(
                chunks, namespace, embedding_model, parsed_document=parsed
            )
            
            return IngestionResult(
                document_id=doc_id,
                source=source_ref,
                chunk_count=len(chunks),
                page_count=parsed.page_count,
                chunks=chunks,
                errors=parsed.parse_errors + processing_errors,
            )
        except Exception as e:
            logger.exception(f"Error ingesting file: {path}")
            return IngestionResult(doc_id, source_ref, errors=[str(e)])

    async def ingest_text(
        self,
        text: str,
        namespace: str = "default",
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        import uuid
        from unified_memory.core.types import PageContent
        
        doc_id = document_id or str(uuid.uuid4())
        source_ref = SourceReference(source_id=doc_id, source_type=SourceType.TEXT_BLOCK)
        
        parsed = ParsedDocument(
            document_id=doc_id,
            source=source_ref,
            title=title,
            pages=[PageContent(page_number=1, document_id=doc_id, text_blocks=[{"text": text}], full_text=text)],
            full_text=text,
            metadata=metadata or {},
        )
        
        try:
            embedding_model = self.embedding_provider.model_id
            chunks = await self.chunker.chunk(parsed, namespace, embedding_model=embedding_model)
            
            processing_errors = await self._process_chunks(
                chunks, namespace, embedding_model, parsed_document=parsed
            )
            
            return IngestionResult(
                document_id=doc_id,
                source=source_ref,
                chunk_count=len(chunks),
                page_count=1,
                chunks=chunks,
                errors=processing_errors,
            )
        except Exception as e:
            logger.exception(f"Error ingesting text: {doc_id}")
            return IngestionResult(doc_id, source_ref, errors=[str(e)])
