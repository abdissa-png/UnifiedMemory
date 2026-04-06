
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from unified_memory.retrieval.rerankers.models import CohereReranker, BGEReranker
from unified_memory.core.types import RetrievalResult

@pytest.fixture
def sample_results():
    return [
        RetrievalResult(id="1", content="Apple", score=0.5, metadata={}, source="test"),
        RetrievalResult(id="2", content="Banana", score=0.4, metadata={}, source="test"),
    ]

@pytest.mark.asyncio
async def test_cohere_reranker(sample_results):
    mock_response = MagicMock()
    mock_result_item0 = MagicMock()
    mock_result_item0.index = 1  # Banana
    mock_result_item0.relevance_score = 0.9

    mock_result_item1 = MagicMock()
    mock_result_item1.index = 0  # Apple
    mock_result_item1.relevance_score = 0.1

    mock_response.results = [mock_result_item0, mock_result_item1]
    mock_response.meta = None

    mock_async_client = AsyncMock()
    mock_async_client.rerank.return_value = mock_response

    with patch.dict('sys.modules', {'cohere': MagicMock()}):
        with patch("cohere.AsyncClient", return_value=mock_async_client):
            reranker = CohereReranker(api_key="fake")
            reranked = await reranker.rerank("fruit", sample_results)

            assert len(reranked) == 2
            assert reranked[0].id == "2"  # Banana first
            assert reranked[0].score == 0.9
            assert reranked[1].id == "1"

@pytest.mark.asyncio
async def test_bge_reranker_import_error():
    # Verify raises ImportError if sentence-transformers missing
    with patch.dict('sys.modules', {'sentence_transformers': None}):
        with pytest.raises(ImportError):
            BGEReranker()

@pytest.mark.asyncio
async def test_bge_reranker_success(sample_results):
    # Mock sentence_transformers and CrossEncoder
    mock_cross_encoder_cls = MagicMock()
    mock_model = mock_cross_encoder_cls.return_value
    # predict returns list of scores
    mock_model.predict.return_value = [0.1, 0.9]
    
    with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
        with patch("sentence_transformers.CrossEncoder", mock_cross_encoder_cls):
            reranker = BGEReranker()
            reranked = await reranker.rerank("fruit", sample_results)
            
            assert len(reranked) == 2
            assert reranked[0].id == "2" # Score 0.9
            assert reranked[0].score == 0.9
            assert reranked[1].id == "1" # Score 0.1
