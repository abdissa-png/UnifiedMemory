"""
Hierarchical Configuration System.

Location: core/config.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 8
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_ELASTICSEARCH_URL = "http://localhost:9200"
DEFAULT_ARTIFACT_DIR = "/tmp/memory_artifacts"


@dataclass
class SystemConfig:
    """Global system-wide defaults."""
    default_text_model: str = "text-embedding-3-small"
    default_vision_model: str = "clip-vit-base-patch32"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    enable_multi_tenant: bool = True


@dataclass
class IngestionConfig:
    """
    Resolved configuration for a specific ingestion request.
    
    Merges settings from:
    1. System Defaults
    2. Tenant Settings (e.g., authoritative embedding model)
    3. Namespace Settings (e.g., specific collections)
    4. Request Options (e.g., high-priority override)
    """
    
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    text_collection: str
    respect_sentence_boundaries: bool = True
    
    @classmethod
    def resolve(
        cls,
        system: SystemConfig,
        tenant_data: Optional[Dict[str, Any]] = None,
        namespace_data: Optional[Dict[str, Any]] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> IngestionConfig:
        """
        Hierarchical resolution logic.
        Higher levels override lower levels.
        """
        request = request_options or {}
        ns = namespace_data or {}
        tenant = tenant_data or {}
        
        # 1. Chunk Size
        chunk_size = (
            request.get("chunk_size") or 
            ns.get("chunk_size") or 
            tenant.get("chunk_size") or 
            system.default_chunk_size
        )
        
        # 2. Chunk Overlap
        chunk_overlap = (
            request.get("chunk_overlap") or 
            ns.get("chunk_overlap") or 
            tenant.get("chunk_overlap") or 
            system.default_chunk_overlap
        )
        
        # 3. Embedding Model (Tenant is usually authoritative for consistency)
        # We prioritize request if explicitly provided, else tenant.
        embedding_model = (
            request.get("embedding_model") or 
            tenant.get("text_embedding", {}).get("model") or 
            system.default_text_model
        )
        
        # 4. Collections
        # IMPORTANT: Keep naming consistent with NamespaceConfig.__post_init__
        # which uses f"{tenant_id}_texts" (no "tenant_" prefix).
        text_collection = ns.get("text_collection") or f"{tenant.get('tenant_id', 'default')}_texts"
        
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            text_collection=text_collection,
            respect_sentence_boundaries=request.get(
                "respect_sentence_boundaries", True
            )
        )


# -----------------------------------------------------------------------------
# AppConfig schema (used to load/validate YAML config)
# -----------------------------------------------------------------------------


@dataclass
class InfraConfig:
    """
    Infrastructure backends and connection URLs.

    These map directly to the keys consumed by SystemContext._build_* helpers.
    """

    kv_store: str = "memory"  # "memory" | "redis"
    redis_url: str = DEFAULT_REDIS_URL

    vector_store: str = "memory"  # "memory" | "qdrant"
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_api_key: Optional[str] = None

    graph_store: str = "networkx"  # "networkx" | "neo4j"
    neo4j_uri: str = DEFAULT_NEO4J_URI
    neo4j_auth_user: str = "neo4j"
    neo4j_auth_password: str = "password"

    sparse_retriever: str = "bm25"  # "bm25" | "elasticsearch"
    elasticsearch_url: str = DEFAULT_ELASTICSEARCH_URL
    elasticsearch_index: str = "unified_memory_content"
    elasticsearch_api_key: Optional[str] = None


@dataclass
class EmbeddingProviderConfig:
    provider: str
    model: str
    dimension: int
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    modality: str = "text"  # "text" | "vision" | "image" | "shared"


@dataclass
class LLMProviderConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    supports_images: Optional[bool] = None


@dataclass
class ExtractorConfig:
    type: str = "mock"  # "mock" | "llm"
    llm_provider: Optional[str] = None


@dataclass
class RerankerConfig:
    type: str  # "bge" | "cohere"
    model: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class DefaultsConfig:
    """
    Application-level defaults that influence ingestion and retrieval.
    """

    retrieval_paths: List[str] = field(
        default_factory=lambda: ["dense", "sparse", "graph"]
    )
    skip_embedding: bool = False
    enable_visual_indexing: bool = False


@dataclass
class AppConfig:
    """
    Full application configuration loaded from YAML.

    This is a higher-level representation; SystemContext still consumes a
    flattened dict produced via app_config_to_dict().
    """

    infra: InfraConfig = field(default_factory=InfraConfig)
    embedding_providers: Dict[str, EmbeddingProviderConfig] = field(
        default_factory=dict
    )
    llm_providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    extractors: Dict[str, ExtractorConfig] = field(default_factory=dict)
    rerankers: Dict[str, RerankerConfig] = field(default_factory=dict)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    config_version: int = 1


def _interpolate_env(obj: Any) -> Any:
    """
    Recursively interpolate ${VAR} in strings using environment variables.
    """
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    if isinstance(obj, str):
        # Simple ${VAR} replacement
        import re

        pattern = re.compile(r"\$\{([^}]+)\}")

        def repl(match: "re.Match[str]") -> str:
            var = match.group(1)
            return os.getenv(var, "")

        return pattern.sub(repl, obj)
    return obj


def load_app_config(path: Path) -> AppConfig:
    """
    Load an AppConfig from a YAML file with ${VAR} interpolation.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - only hit when YAML loading is used
        raise ImportError(
            "The 'pyyaml' package is required to load YAML configs. "
            "Install it with: pip install pyyaml"
        ) from exc

    raw = path.read_text()
    data = yaml.safe_load(raw) or {}
    data = _interpolate_env(data)

    infra = InfraConfig(**data.get("infra", {}))

    def _build_mapping(raw_map: Dict[str, Any], cls) -> Dict[str, Any]:
        return {k: cls(**v) for k, v in raw_map.items()}

    embedding_providers = _build_mapping(
        data.get("embedding_providers", {}), EmbeddingProviderConfig
    )
    llm_providers = _build_mapping(
        data.get("llm_providers", {}), LLMProviderConfig
    )
    extractors = _build_mapping(
        data.get("extractors", {}), ExtractorConfig
    )
    rerankers = _build_mapping(
        data.get("rerankers", {}), RerankerConfig
    )
    defaults = DefaultsConfig(**data.get("defaults", {}))
    version = int(data.get("config_version", 1))

    return AppConfig(
        infra=infra,
        embedding_providers=embedding_providers,
        llm_providers=llm_providers,
        extractors=extractors,
        rerankers=rerankers,
        defaults=defaults,
        config_version=version,
    )


def app_config_to_dict(app: AppConfig) -> Dict[str, Any]:
    """
    Flatten AppConfig into the dict shape expected by SystemContext.__init__.
    """
    cfg: Dict[str, Any] = {}

    infra = app.infra
    cfg.update(
        {
            "kv_store": infra.kv_store,
            "redis_url": infra.redis_url,
            "vector_store": infra.vector_store,
            "qdrant_url": infra.qdrant_url,
            "qdrant_api_key": infra.qdrant_api_key,
            "graph_store": infra.graph_store,
            "neo4j_uri": infra.neo4j_uri,
            "neo4j_auth": (infra.neo4j_auth_user, infra.neo4j_auth_password),
            "sparse_retriever": infra.sparse_retriever,
            "elasticsearch_url": infra.elasticsearch_url,
            "elasticsearch_index": infra.elasticsearch_index,
            "elasticsearch_api_key": infra.elasticsearch_api_key,
        }
    )

    cfg["embedding_providers"] = {
        key: asdict(val) for key, val in app.embedding_providers.items()
    }
    cfg["llm_providers"] = {
        key: asdict(val) for key, val in app.llm_providers.items()
    }
    cfg["extractors"] = {
        key: asdict(val) for key, val in app.extractors.items()
    }
    cfg["rerankers"] = {
        key: asdict(val) for key, val in app.rerankers.items()
    }

    cfg["defaults"] = asdict(app.defaults)
    cfg["config_version"] = app.config_version
    return cfg


def validate_config_compatibility(app: AppConfig) -> List[str]:
    """
    Validate an AppConfig and return a list of human-readable errors.

    This focuses on combinations that are known to cause runtime errors:
    - Graph retrieval with no graph store
    - Elasticsearch sparse retriever without URL
    - Qdrant vector store without URL
    - Redis KV without URL
    - Visual indexing enabled without any vision-capable embedding provider
    - skip_embedding default combined with graph retrieval path
    """
    errors: List[str] = []
    infra = app.infra
    defaults = app.defaults

    # Graph retrieval requires a graph store
    if "graph" in defaults.retrieval_paths and infra.graph_store == "none":
        errors.append(
            "defaults.retrieval_paths includes 'graph' but infra.graph_store is 'none'. "
            "Either disable graph retrieval or configure a graph store."
        )

    # Elasticsearch sparse retriever requires URL (and index when sparse path is enabled)
    if infra.sparse_retriever == "elasticsearch":
        if not infra.elasticsearch_url:
            errors.append(
                "infra.sparse_retriever is 'elasticsearch' but elasticsearch_url is empty."
            )
        if "sparse" in defaults.retrieval_paths and not infra.elasticsearch_index:
            errors.append(
                "defaults.retrieval_paths includes 'sparse' with "
                "infra.sparse_retriever='elasticsearch' but elasticsearch_index is empty."
            )

    # Qdrant vector store requires URL
    if infra.vector_store == "qdrant":
        if not infra.qdrant_url:
            errors.append(
                "infra.vector_store is 'qdrant' but qdrant_url is empty."
            )

    # Redis KV requires URL
    if infra.kv_store == "redis":
        if not infra.redis_url:
            errors.append(
                "infra.kv_store is 'redis' but redis_url is empty."
            )

    # Visual indexing requires at least one vision-capable embedding provider
    if defaults.enable_visual_indexing:
        has_vision = any(
            ep.modality.lower() in ("vision", "image", "shared")
            for ep in app.embedding_providers.values()
        )
        if not has_vision:
            errors.append(
                "defaults.enable_visual_indexing is True but no embedding_providers "
                "entry has modality 'vision', 'image', or 'shared'."
            )

    # skip_embedding default is incompatible with graph retrieval
    if defaults.skip_embedding and "graph" in defaults.retrieval_paths:
        errors.append(
            "defaults.skip_embedding=True while 'graph' is in defaults.retrieval_paths. "
            "Graph retrieval relies on entity/relation embeddings."
        )

    # Dense / graph retrieval requires at least one embedding provider to be
    # configured at the application level.  Tenant configs may still be
    # misconfigured, but an entirely empty embedding_providers map is almost
    # always a wiring error when dense/graph are enabled.
    if (
        any(path in defaults.retrieval_paths for path in ("dense", "graph"))
        and not app.embedding_providers
    ):
        errors.append(
            "defaults.retrieval_paths includes 'dense' or 'graph' but "
            "embedding_providers is empty. Configure at least one embedding "
            "provider or disable these retrieval paths."
        )

    # LLM extractors must reference an existing LLM provider key.
    for ext_key, ext_cfg in app.extractors.items():
        if ext_cfg.type == "llm" and ext_cfg.llm_provider:
            if ext_cfg.llm_provider not in app.llm_providers:
                errors.append(
                    f"Extractor '{ext_key}' references llm_provider "
                    f"'{ext_cfg.llm_provider}' which is not defined in llm_providers."
                )

    return errors
