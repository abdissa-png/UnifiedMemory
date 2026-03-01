
import pytest
import asyncio
from unified_memory.retrieval.sparse_bm25 import BM25SparseRetriever

@pytest.mark.asyncio
async def test_bm25_indexing_and_retrieval():
    retriever = BM25SparseRetriever()
    namespace = "test_ns"
    
    documents = [
        {"id": "doc1", "content": "The quick brown fox jumps over the lazy dog.", "metadata": {"category": "fox"}},
        {"id": "doc2", "content": "The dog chases the cat.", "metadata": {"category": "dog"}},
        {"id": "doc3", "content": "Python is a programming language.", "metadata": {"category": "tech"}},
    ]
    
    # 1. Index
    count = await retriever.index(documents, namespace)
    assert count == 3
    
    # 2. Retrieve
    # Query "fox" -> matches doc1
    results = await retriever.retrieve("fox", namespace)
    assert len(results) == 1
    assert results[0].id == "doc1"
    
    # Query "dog" -> matches doc1 and doc2
    results = await retriever.retrieve("dog", namespace)
    assert len(results) == 2
    # doc2 likely higher score as "dog" is more significant in shorter text? BM25 logic dependent.
    # Just check ids exist.
    ids = [r.id for r in results]
    assert "doc1" in ids
    assert "doc2" in ids
    
    # 3. Filter
    results = await retriever.retrieve("dog", namespace, filters={"category": "dog"})
    assert len(results) == 1
    assert results[0].id == "doc2"

@pytest.mark.asyncio
async def test_bm25_empty_and_unknown_namespace():
    retriever = BM25SparseRetriever()
    
    # Unknown namespace
    results = await retriever.retrieve("query", "unknown")
    assert results == []
    
    # Index empty
    count = await retriever.index([], "created_but_empty")
    assert count == 0
    # Retrieve from empty
    results = await retriever.retrieve("query", "created_but_empty")
    assert results == []

@pytest.mark.asyncio
async def test_bm25_namespace_isolation():
    retriever = BM25SparseRetriever()
    
    # Index "fox" in ns1
    await retriever.index([{"id": "doc1", "content": "The quick brown fox"}], "ns1")
    # Index "cat" in ns2
    await retriever.index([{"id": "doc2", "content": "The quick black cat"}], "ns2")
    
    # Query ns1 for "cat" -> should be empty
    results = await retriever.retrieve("cat", "ns1")
    assert results == []
    
    # Query ns2 for "fox" -> should be empty
    results = await retriever.retrieve("fox", "ns2")
    assert results == []
    
    # Query ns1 for "fox" -> should have 1
    results = await retriever.retrieve("fox", "ns1")
    assert len(results) == 1
    assert results[0].id == "doc1"

@pytest.mark.asyncio
async def test_bm25_unicode_content():
    retriever = BM25SparseRetriever()
    namespace = "unicode_ns"
    
    # Japanese and Emoji
    documents = [
        {"id": "jp1", "content": "こんにちは、世界！", "metadata": {}},
        {"id": "emoji1", "content": "The robot 🤖 is happy 😃.", "metadata": {}},
    ]
    
    await retriever.index(documents, namespace)
    
    # Query Japanese
    results = await retriever.retrieve("こんにちは", namespace)
    assert len(results) == 1
    assert results[0].id == "jp1"
    
    # Query Emoji
    results = await retriever.retrieve("🤖", namespace)
    assert len(results) == 1
    assert results[0].id == "emoji1"

@pytest.mark.asyncio
async def test_bm25_incremental_indexing():
    retriever = BM25SparseRetriever()
    namespace = "inc_ns"
    
    # First batch
    await retriever.index([{"id": "doc1", "content": "The quick brown fox"}], namespace)
    results = await retriever.retrieve("fox", namespace)
    assert len(results) == 1
    
    # Second batch (should append, not overwrite?)
    # Design choice: BM25 typically appends/re-indexes.
    # Our implementation uses a dictionary internally, so it appends by id.
    await retriever.index([{"id": "doc2", "content": "The black cat"}], namespace)
    
    # Verify BOTH are searchable
    results1 = await retriever.retrieve("fox", namespace)
    assert len(results1) == 1
    results2 = await retriever.retrieve("cat", namespace)
    assert len(results2) == 1
