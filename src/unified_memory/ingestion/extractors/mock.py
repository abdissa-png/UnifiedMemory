import re
from typing import List
from unified_memory.core.types import Chunk
from unified_memory.ingestion.extractors.base import Extractor
from unified_memory.ingestion.extractors.schema import ExtractionResult, ExtractedEntity, ExtractedRelation

class MockExtractor(Extractor):
    """Simple regex-based extractor for testing."""
    
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        result = ExtractionResult()
        
        # Simple entity extraction: capitalized words as "Entity"
        # (This is just for testing the pipeline flow)
        entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', chunk.content)
        for name in set(entities):
            result.entities.append(ExtractedEntity(name=name, type="Concept"))
            
        # Mock relation if more than one entity
        if len(result.entities) >= 2:
            result.relations.append(ExtractedRelation(
                source_entity=result.entities[0].name,
                target_entity=result.entities[1].name,
                relation_type="RELATED_TO"
            ))
            
        return result
