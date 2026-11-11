import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.services.rag_service import RagService, RetrievalBranchResult


def _vector_chunk(chunk_id: str = "chunk-1", kb_id: str = "kb-a") -> dict:
    return {
        "kb_id": kb_id,
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "title": "title",
        "content": "content",
        "score": 0.9,
        "metadata": {},
    }


def _bm25_chunk(chunk_id: str = "chunk-2", kb_id: str = "kb-b") -> dict:
    return {
        "kb_id": kb_id,
        "document_id": "doc-2",
        "chunk_id": chunk_id,
        "title": "title",
        "content": "content",
        "bm25_score": 1.2,
        "metadata": {},
    }


@pytest.fixture
def rag_service() -> RagService:
    service = RagService(MagicMock())
    service._get_vector_top_k = MagicMock(return_value=20)
    service._get_bm25_top_k = MagicMock(return_value=20)
    service._get_rrf_k = MagicMock(return_value=60)
    return service


def _scope() -> tuple[list[MagicMock], dict[str, str]]:
    knowledge_bases = [
        MagicMock(id="kb-a", name="A"),
        MagicMock(id="kb-b", name="B"),
    ]
    return knowledge_bases, {"kb-a": "v1", "kb-b": "v2"}


def test_global_hybrid_degrades_to_bm25_when_vector_fails(rag_service: RagService):
    knowledge_bases, versions = _scope()
    rag_service._run_vector_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(source="vector", error=RuntimeError("down"))
    )
    rag_service._run_bm25_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(
            source="bm25",
            chunks=[_bm25_chunk()],
            details={"bm25_request_count": 1},
        )
    )

    chunks, metadata = asyncio.run(
        rag_service._retrieve_multi_kb(
            tenant_id="tenant-a",
            request=MagicMock(retrieval_options=None),
            knowledge_bases=knowledge_bases,
            search_query="query",
            mode="hybrid",
            top_k=10,
            index_versions=versions,
        )
    )

    assert [item["chunk_id"] for item in chunks] == ["chunk-2"]
    assert metadata["degraded"] is True
    assert metadata["degraded_reason"] == "vector search failed"
    assert metadata["fusion_scope"] == "global"


def test_global_hybrid_degrades_to_vector_when_bm25_fails(rag_service: RagService):
    knowledge_bases, versions = _scope()
    rag_service._run_vector_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(
            source="vector",
            chunks=[_vector_chunk()],
            details={"embedding_request_count": 1, "vector_request_count": 1},
        )
    )
    rag_service._run_bm25_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(source="bm25", error=RuntimeError("down"))
    )

    chunks, metadata = asyncio.run(
        rag_service._retrieve_multi_kb(
            tenant_id="tenant-a",
            request=MagicMock(retrieval_options=None),
            knowledge_bases=knowledge_bases,
            search_query="query",
            mode="hybrid",
            top_k=10,
            index_versions=versions,
        )
    )

    assert [item["chunk_id"] for item in chunks] == ["chunk-1"]
    assert metadata["degraded_reason"] == "bm25 search failed"
    assert metadata["embedding_request_count"] == 1
    assert metadata["vector_request_count"] == 1


def test_global_hybrid_raises_when_both_branches_fail(rag_service: RagService):
    knowledge_bases, versions = _scope()
    rag_service._run_vector_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(source="vector", error=RuntimeError("down"))
    )
    rag_service._run_bm25_scope_branch = AsyncMock(
        return_value=RetrievalBranchResult(source="bm25", error=RuntimeError("down"))
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(
            rag_service._retrieve_multi_kb(
                tenant_id="tenant-a",
                request=MagicMock(retrieval_options=None),
                knowledge_bases=knowledge_bases,
                search_query="query",
                mode="hybrid",
                top_k=10,
                index_versions=versions,
            )
        )

    assert exc_info.value.code == ErrorCode.RETRIEVAL_FAILED.code


def test_global_hybrid_starts_both_branches_before_either_finishes(rag_service: RagService):
    knowledge_bases, versions = _scope()
    started: set[str] = set()
    release = asyncio.Event()

    async def vector(**_kwargs):
        started.add("vector")
        if started == {"vector", "bm25"}:
            release.set()
        await release.wait()
        return RetrievalBranchResult(source="vector", chunks=[_vector_chunk()])

    async def bm25(**_kwargs):
        started.add("bm25")
        if started == {"vector", "bm25"}:
            release.set()
        await release.wait()
        return RetrievalBranchResult(source="bm25", chunks=[_bm25_chunk()])

    rag_service._run_vector_scope_branch = vector
    rag_service._run_bm25_scope_branch = bm25

    asyncio.run(
        rag_service._retrieve_multi_kb(
            tenant_id="tenant-a",
            request=MagicMock(retrieval_options=None),
            knowledge_bases=knowledge_bases,
            search_query="query",
            mode="hybrid",
            top_k=10,
            index_versions=versions,
        )
    )

    assert started == {"vector", "bm25"}


def test_vector_scope_embeds_once_and_queries_all_kbs_once(rag_service: RagService):
    rag_service.embedding_provider.embed_query = AsyncMock(return_value=[0.1, 0.2])
    rag_service.vector_store.similarity_search_scope = AsyncMock(
        return_value=[_vector_chunk()]
    )
    _, versions = _scope()

    result = asyncio.run(
        rag_service._run_vector_scope_branch(
            tenant_id="tenant-a",
            index_versions=versions,
            search_query="query",
            top_k=20,
        )
    )

    rag_service.embedding_provider.embed_query.assert_awaited_once_with("query")
    rag_service.vector_store.similarity_search_scope.assert_awaited_once_with(
        [0.1, 0.2],
        tenant_id="tenant-a",
        index_versions=versions,
        top_k=20,
    )
    assert result.details["embedding_request_count"] == 1
    assert result.details["vector_request_count"] == 1


def test_bm25_scope_queries_all_kbs_once(rag_service: RagService):
    provider = MagicMock()
    provider.keyword_search_scope = AsyncMock(return_value=[_bm25_chunk()])
    rag_service._build_keyword_search_provider = MagicMock(return_value=provider)
    _, versions = _scope()

    result = asyncio.run(
        rag_service._run_bm25_scope_branch(
            tenant_id="tenant-a",
            index_versions=versions,
            search_query="query",
            top_k=20,
        )
    )

    provider.keyword_search_scope.assert_awaited_once_with(
        query="query",
        tenant_id="tenant-a",
        index_versions=versions,
        top_k=20,
    )
    assert result.details["bm25_request_count"] == 1


def test_resolve_empty_reason_no_indexed_chunks(rag_service: RagService):
    rag_service.chunk_repository.count_by_kb_id = MagicMock(return_value=0)
    assert rag_service._resolve_empty_reason("tenant-a", ["kb-a"], []) == "no_indexed_chunks"


def test_resolve_empty_reason_no_chunks_matched(rag_service: RagService):
    rag_service.chunk_repository.count_by_kb_id = MagicMock(return_value=3)
    assert rag_service._resolve_empty_reason("tenant-a", ["kb-a"], []) == "no_chunks_matched"
