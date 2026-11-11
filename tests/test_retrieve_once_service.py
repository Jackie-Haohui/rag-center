import asyncio
from types import SimpleNamespace

from app.schemas.rag import (
    QueryOptions,
    RetrieveContextMetadata,
    RetrieveGraphInjectionMetadata,
    RetrieveRequest,
    RetrieveRerankMetadata,
)
from app.services.retrieve_once_service import RetrieveOnceService


def _chunk(chunk_id: str = "chunk-1") -> dict:
    return {
        "kb_id": "kb",
        "document_id": "doc",
        "chunk_id": chunk_id,
        "title": "title",
        "content": "content",
        "score": 1.0,
        "metadata": {},
    }


class FakeBackend:
    def __init__(self):
        self.calls: list[str] = []
        self.rerank_query: str | None = None
        self.graph_candidate_expansion_service = SimpleNamespace(expand_candidates=self._graph)
        self.context_expansion_service = SimpleNamespace(expand=self._context)

    def _get_retrieval_mode(self, request):
        return "hybrid"

    def _get_retrieval_candidate_k(self, request, top_k):
        return 20

    async def _retrieve_multi_kb(self, **kwargs):
        self.calls.append("retrieve")
        return [_chunk()], {"mode": "hybrid", "degraded": False}

    def _resolve_empty_reason(self, *args, **kwargs):
        return None

    async def _graph(self, **kwargs):
        self.calls.append("graph")
        return SimpleNamespace(
            candidates=kwargs["seeds"],
            metadata=RetrieveGraphInjectionMetadata(executed=True, candidate_count=1),
        )

    async def _maybe_rerank(self, *, request, rerank_query, chunks, top_k):
        self.calls.append("rerank")
        self.rerank_query = rerank_query
        return chunks[:top_k], RetrieveRerankMetadata(
            enabled=True,
            provider="fake",
            candidate_count=len(chunks),
        )

    async def _context(self, **kwargs):
        self.calls.append("context")
        return SimpleNamespace(
            anchors=kwargs["anchors"],
            metadata=RetrieveContextMetadata(executed=True, anchor_count=1),
        )


def test_retrieve_once_runs_shared_pipeline_without_orchestration_side_effects():
    backend = FakeBackend()
    request = RetrieveRequest(
        kb_id="kb",
        user_id="user",
        query="raw question",
        profile="quality",
        top_k=5,
        query_options=QueryOptions(enabled=True, strategy="rewrite"),
    )
    result = asyncio.run(
        RetrieveOnceService(backend).execute(
            tenant_id="tenant",
            request=request,
            knowledge_bases=[SimpleNamespace(id="kb", name="KB")],
            index_versions={"kb": "v2"},
            search_query="planned query",
            query_id="Q1",
            aspect_ids=["A1"],
            round=1,
        )
    )

    assert backend.calls == ["retrieve", "graph", "rerank", "context"]
    assert backend.rerank_query == "planned query"
    assert result.query == "planned query"
    assert result.query_id == "Q1"
    assert result.aspect_ids == ["A1"]
    assert result.round == 1
    assert result.candidate_snapshot[0]["chunk_id"] == "chunk-1"
    assert result.retrieved_chunks[0]["chunk_id"] == "chunk-1"
    assert result.degraded is False


def test_retrieve_once_keeps_raw_rerank_query_when_rewrite_is_disabled():
    backend = FakeBackend()
    request = RetrieveRequest(
        kb_id="kb",
        user_id="user",
        query="raw research question",
        profile="quality",
        top_k=5,
        query_options=QueryOptions(enabled=False, strategy="noop"),
    )

    asyncio.run(
        RetrieveOnceService(backend).execute(
            tenant_id="tenant",
            request=request,
            knowledge_bases=[SimpleNamespace(id="kb", name="KB")],
            index_versions={"kb": "v2"},
            search_query="planned research query",
            graph_injection=False,
            expand_context=False,
        )
    )

    assert backend.rerank_query == "raw research question"
