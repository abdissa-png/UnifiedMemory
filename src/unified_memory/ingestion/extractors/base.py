from abc import ABC, abstractmethod
from typing import Any, List, Optional
from unified_memory.core.types import Chunk
from unified_memory.ingestion.extractors.schema import ExtractionResult

class Extractor(ABC):
    """Base class for all schema/entity extractors."""
    
    @abstractmethod
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relations from a chunk."""
        raise NotImplementedError
