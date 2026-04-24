"""Helpers for tenant registration forms in the Streamlit demo."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


_DEFAULT_BOOL_OPTION = "Use backend default"


def render_tenant_config_inputs(prefix: str) -> Dict[str, Any]:
    """Render optional tenant-config fields and return raw form values."""
    with st.expander("Tenant Config (Recommended)", expanded=False):
        st.caption(
            "Use explicit provider/model pairs that exist in your app configuration. "
            "If you leave fields blank, backend defaults are used."
        )

        text_embedding_provider = st.text_input(
            "Text Embedding Provider",
            key=f"{prefix}_text_embedding_provider",
            placeholder="openai",
        )
        text_embedding_model = st.text_input(
            "Text Embedding Model",
            key=f"{prefix}_text_embedding_model",
            placeholder="text-embedding-3-small",
        )
        text_embedding_dimension = st.text_input(
            "Text Embedding Dimension",
            key=f"{prefix}_text_embedding_dimension",
            placeholder="1536",
        )

        vision_embedding_provider = st.text_input(
            "Vision Embedding Provider",
            key=f"{prefix}_vision_embedding_provider",
            placeholder="openai",
        )
        vision_embedding_model = st.text_input(
            "Vision Embedding Model",
            key=f"{prefix}_vision_embedding_model",
            placeholder="clip-vit-base-patch32",
        )
        vision_embedding_dimension = st.text_input(
            "Vision Embedding Dimension",
            key=f"{prefix}_vision_embedding_dimension",
            placeholder="512",
        )

        llm_provider = st.text_input(
            "LLM Provider",
            key=f"{prefix}_llm_provider",
            placeholder="openai",
        )
        llm_model = st.text_input(
            "LLM Model",
            key=f"{prefix}_llm_model",
            placeholder="gpt-4o-mini",
        )

        chunk_size = st.text_input(
            "Chunk Size",
            key=f"{prefix}_chunk_size",
            placeholder="512",
        )
        chunk_overlap = st.text_input(
            "Chunk Overlap",
            key=f"{prefix}_chunk_overlap",
            placeholder="64",
        )
        batch_size = st.text_input(
            "Batch Size",
            key=f"{prefix}_batch_size",
            placeholder="100",
        )
        chunker_type = st.selectbox(
            "Chunker Type",
            ["Use backend default", "fixed_size", "recursive", "semantic"],
            key=f"{prefix}_chunker_type",
        )

        enable_graph_storage = _render_bool_override("Enable Graph Storage", prefix, "graph_storage")
        enable_visual_indexing = _render_bool_override("Enable Visual Indexing", prefix, "visual_indexing")
        enable_entity_extraction = _render_bool_override("Enable Entity Extraction", prefix, "entity_extraction")
        enable_relation_extraction = _render_bool_override("Enable Relation Extraction", prefix, "relation_extraction")
        deduplication_enabled = _render_bool_override("Enable Deduplication", prefix, "deduplication")

        st.markdown("#### Extraction Config")
        extraction_extractor_type = st.text_input(
            "Extractor Type",
            key=f"{prefix}_extraction_extractor_type",
            placeholder="llm",
            help="Extractor key/type used for graph extraction (e.g. llm, llm-default, mock).",
        )
        extraction_llm_model = st.text_input(
            "Extraction LLM Model (optional)",
            key=f"{prefix}_extraction_llm_model",
            placeholder="meta-llama/Llama-3.3-70B-Instruct",
        )
        extraction_entity_types = st.text_input(
            "Allowed Entity Types (comma-separated, optional)",
            key=f"{prefix}_extraction_entity_types",
            placeholder="person, organization, location",
        )
        extraction_relation_types = st.text_input(
            "Allowed Relation Types (comma-separated, optional)",
            key=f"{prefix}_extraction_relation_types",
            placeholder="works_for, located_in",
        )
        extraction_confidence_threshold = st.text_input(
            "Extraction Confidence Threshold (optional)",
            key=f"{prefix}_extraction_confidence_threshold",
            placeholder="0.5",
        )
        extraction_batch_size = st.text_input(
            "Extraction Batch Size (optional)",
            key=f"{prefix}_extraction_batch_size",
            placeholder="10",
        )
        extraction_strict_type_filtering = _render_bool_override(
            "Strict Type Filtering",
            prefix,
            "extraction_strict_type_filtering",
        )

    return {
        "text_embedding_provider": text_embedding_provider,
        "text_embedding_model": text_embedding_model,
        "text_embedding_dimension": text_embedding_dimension,
        "vision_embedding_provider": vision_embedding_provider,
        "vision_embedding_model": vision_embedding_model,
        "vision_embedding_dimension": vision_embedding_dimension,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "batch_size": batch_size,
        "chunker_type": chunker_type,
        "enable_graph_storage": enable_graph_storage,
        "enable_visual_indexing": enable_visual_indexing,
        "enable_entity_extraction": enable_entity_extraction,
        "enable_relation_extraction": enable_relation_extraction,
        "deduplication_enabled": deduplication_enabled,
        "extraction_extractor_type": extraction_extractor_type,
        "extraction_llm_model": extraction_llm_model,
        "extraction_entity_types": extraction_entity_types,
        "extraction_relation_types": extraction_relation_types,
        "extraction_confidence_threshold": extraction_confidence_threshold,
        "extraction_batch_size": extraction_batch_size,
        "extraction_strict_type_filtering": extraction_strict_type_filtering,
    }


def build_tenant_config_overrides(raw_values: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw form values into API payload overrides."""
    overrides: Dict[str, Any] = {}
    for key in (
        "text_embedding_provider",
        "text_embedding_model",
        "vision_embedding_provider",
        "vision_embedding_model",
        "llm_provider",
        "llm_model",
        "extraction_extractor_type",
        "extraction_llm_model",
    ):
        value = str(raw_values.get(key, "")).strip()
        if value:
            overrides[key] = value

    for key in ("text_embedding_dimension", "vision_embedding_dimension", "chunk_size", "chunk_overlap", "batch_size"):
        parsed = _parse_optional_int(raw_values.get(key), key)
        if parsed is not None:
            overrides[key] = parsed

    chunker_type = raw_values.get("chunker_type")
    if chunker_type and chunker_type != "Use backend default":
        overrides["chunker_type"] = chunker_type

    for key in (
        "enable_graph_storage",
        "enable_visual_indexing",
        "enable_entity_extraction",
        "enable_relation_extraction",
        "deduplication_enabled",
        "extraction_strict_type_filtering",
    ):
        parsed_bool = _parse_optional_bool(raw_values.get(key))
        if parsed_bool is not None:
            overrides[key] = parsed_bool

    parsed_extraction_batch = _parse_optional_int(
        raw_values.get("extraction_batch_size"),
        "extraction_batch_size",
    )
    if parsed_extraction_batch is not None:
        overrides["extraction_batch_size"] = parsed_extraction_batch

    parsed_confidence = _parse_optional_float(
        raw_values.get("extraction_confidence_threshold"),
        "extraction_confidence_threshold",
    )
    if parsed_confidence is not None:
        overrides["extraction_confidence_threshold"] = parsed_confidence

    for raw_key, out_key in (
        ("extraction_entity_types", "extraction_entity_types"),
        ("extraction_relation_types", "extraction_relation_types"),
    ):
        parsed_list = _parse_optional_csv(raw_values.get(raw_key))
        if parsed_list is not None:
            overrides[out_key] = parsed_list

    return overrides


def _render_bool_override(label: str, prefix: str, suffix: str) -> str:
    return st.selectbox(
        label,
        [_DEFAULT_BOOL_OPTION, "True", "False"],
        key=f"{prefix}_{suffix}",
    )


def _parse_optional_int(value: Any, field_name: str) -> Optional[int]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _parse_optional_float(value: Any, field_name: str) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a float") from exc


def _parse_optional_csv(value: Any) -> Optional[list[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]
