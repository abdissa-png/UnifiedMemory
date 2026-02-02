import pytest
import asyncio
from typing import AsyncGenerator



@pytest.fixture
async def mock_embedding_model():
    """Mock embedding model config."""
    return {
        "provider": "test",
        "model": "test-model",
        "dimension": 4
    }
