"""
Retrieval base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from unified_memory.core.types import Memory

class Retriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        namespace: str,
        limit: int = 10,
        **kwargs: Any
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        raise NotImplementedError
