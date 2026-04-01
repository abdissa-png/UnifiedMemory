"""
LLM-based Entity & Relation Extractor.

Uses a BaseLLMProvider to extract structured entities and relations
from text chunks via prompt engineering and JSON parsing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from unified_memory.core.types import Chunk
from unified_memory.ingestion.extractors.base import Extractor
from unified_memory.ingestion.extractors.schema import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
)
from unified_memory.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are a knowledge-graph extraction engine.

Given the following text, extract all entities and relationships.

TEXT:
{text}

Return a JSON object with exactly two keys:
- "entities": a list of objects with keys "name", "type", "description"
- "relations": a list of objects with keys "source_entity", "source_type", \
"target_entity", "target_type", "relation_type", "description"

Rules:
- Entity names should be proper nouns or key concepts.
- Entity "type" is a short PascalCase label (e.g. Person, Organization, Concept).
- Relation types should be short, uppercase, underscore-separated verbs (e.g. WORKS_FOR, CREATED_BY).
- "source_type" and "target_type" in a relation should match the "type" of the \
corresponding entity entry when possible.
- Include a brief description for each entity and relation when possible.
- If no entities or relations are found, return empty lists.
- Return ONLY the JSON object, no markdown fencing or extra text.
"""


class LLMExtractor(Extractor):
    """Extractor that delegates to an LLM for entity/relation extraction."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        max_output_tokens: int = 2048,
    ) -> None:
        self._llm = llm_provider
        self._max_output_tokens = max_output_tokens

    async def extract(self, chunk: Chunk) -> ExtractionResult:
        prompt = _EXTRACTION_PROMPT.format(text=chunk.content)

        try:
            raw = await self._llm.generate_structured(
                prompt,
                max_output_tokens=self._max_output_tokens,
                temperature=0.0,
            )
            data = self._parse_json(raw)
        except Exception:
            logger.exception("LLM extraction failed for chunk %s", chunk.chunk_index)
            return ExtractionResult()

        entities: List[ExtractedEntity] = []
        for ent in data.get("entities", []):
            if not ent.get("name"):
                continue
            entities.append(
                ExtractedEntity(
                    name=ent["name"],
                    type=ent.get("type", "Concept"),
                    description=ent.get("description"),
                    properties=ent.get("properties", {}),
                )
            )

        relations: List[ExtractedRelation] = []
        for rel in data.get("relations", []):
            src = rel.get("source_entity", "")
            tgt = rel.get("target_entity", "")
            rtype = rel.get("relation_type", "RELATED_TO")
            if not src or not tgt:
                continue
            relations.append(
                ExtractedRelation(
                    source_entity=src,
                    target_entity=tgt,
                    relation_type=rtype,
                    description=rel.get("description"),
                    confidence=rel.get("confidence", 1.0),
                    source_type=rel.get("source_type") or None,
                    target_type=rel.get("target_type") or None,
                )
            )

        return ExtractionResult(entities=entities, relations=relations)

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Robust JSON parsing using json-repair."""
        from unified_memory.core.json_utils import validate_and_repair_json
        return validate_and_repair_json(raw, expected_keys=["entities", "relations"])
