import pytest
import asyncio
from typing import AsyncGenerator

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_embedding_model():
    """Mock embedding model config."""
    return {
        "provider": "test",
        "model": "test-model",
        "dimension": 4
    }
