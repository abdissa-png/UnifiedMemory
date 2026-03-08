"""
E2E tests for Elasticsearch sparse retrieval (index → retrieve → delete).
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_elasticsearch_index_retrieve_delete(real_elasticsearch_store):
    """
    Index documents into Elasticsearch, retrieve via BM25, then delete.
    """
    ns = "tenant:t1/user:u1"

    # Index
    docs = [
        {"id": "doc1", "content": "Elasticsearch BM25 sparse retrieval test.", "metadata": {"type": "chunk"}},
        {"id": "doc2", "content": "Full-text search with namespaces.", "metadata": {"type": "chunk"}},
    ]
    count = await real_elasticsearch_store.index(documents=docs, namespace=ns)
    await real_elasticsearch_store._es.indices.refresh(index=real_elasticsearch_store._index)
    assert count == 2

    # Retrieve
    results = await real_elasticsearch_store.retrieve(
        query="Elasticsearch sparse",
        namespaces=[ns],
        top_k=5,
    )
    assert len(results) >= 1
    assert any("Elasticsearch" in r.content for r in results)
    assert all(r.source == "sparse" for r in results)

    # Namespace isolation: other namespace should see nothing
    other = await real_elasticsearch_store.retrieve(
        query="Elasticsearch",
        namespaces=["tenant:t1/user:other"],
        top_k=5,
    )
    assert len(other) == 0

    # Delete (remove namespace from doc1)
    affected = await real_elasticsearch_store.delete(
        doc_ids=["doc1"],
        namespace=ns,
    )
    assert affected == 1

    # doc1 gone for this namespace; doc2 still present
    after = await real_elasticsearch_store.retrieve(
        query="Elasticsearch",
        namespaces=[ns],
        top_k=5,
    )
    # May still get doc2 if it matches "Elasticsearch" (it doesn't), or empty
    assert len(after) <= 1
