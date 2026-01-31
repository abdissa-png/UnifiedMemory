"""
Memory module base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from unified_memory.core.types import Memory

class MemoryManager(ABC):
    """Abstract base class for high-level memory management."""
    
    @abstractmethod
    async def add_memory(
        self, 
        content: str, 
        namespace: str,
        **kwargs: Any
    ) -> Memory:
        """Add a new memory."""
        raise NotImplementedError
        
    @abstractmethod
    async def query_memories(
        self, 
        query: str, 
        namespace: str,
        limit: int = 10,
        **kwargs: Any
    ) -> List[Memory]:
        """Query existing memories."""
        raise NotImplementedError
