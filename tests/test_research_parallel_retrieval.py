import asyncio
from time import perf_counter

from app.schemas.rag import (
    RetrieveContextMetadata,
    RetrieveGraphInjectionMetadata,
    RetrieveRequest,
    RetrieveRerankMetadata,
)
from app.schemas.research import PlannedQuery, ResearchState
from app.services.research_parallel_retrieval_service import (
    ResearchParallelRetrievalService,
)
from app.services.retrieve_once_service import RetrieveOnceResult


def _state():
    return ResearchState(
        research_id="research-1",
        original_query="原问题",
        tenant_id="tenant-1",
        kb_ids=["kb-1"],
        index_versions={"kb-1": "v2"},
    )


def _result(query_id, query, aspect_ids, round, latency_ms=30):
    chunk = {
        "chunk_id": f"chunk-{query_id}",
        "kb_id": "kb-1",
        "document_id": "doc-1",
        "index_version": "v2",
        "title": "规则",
        "content": query,
    }
    return RetrieveOnceResult(
        query_id=query_id,
        query=query,
        aspect_ids=aspect_ids,
        round=round,
        candidate_snapshot=[chunk],
        retrieved_chunks=[chunk],
        retrieval_metadata={},
        candidate_filter_metadata={},
        graph_injection_metadata=RetrieveGraphInjectionMetadata(),
        context_metadata=RetrieveContextMetadata(),
        rerank_metadata=RetrieveRerankMetadata(enabled=False, provider="none"),
        latency_ms=latency_ms,
    )


class FakeRetrieveOnce:
    def __init__(self, failed_query_id: str | None = None):
        self.failed_query_id = failed_query_id
        self.active = 0
        self.max_active = 0
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.03)
            if kwargs["query_id"] == self.failed_query_id:
                raise RuntimeError("retrieval failed")
            return _result(
                kwargs["query_id"],
                kwargs["search_query"],
                kwargs["aspect_ids"],
                kwargs["round"],
            )
        finally:
            self.active -= 1


def _queries():
    return [
        PlannedQuery(query_id=f"Q{index}", query=f"查询{index}", aspect_ids=[f"A{index}"])
        for index in range(1, 4)
    ]


def test_initial_queries_run_concurrently_with_shared_scope_and_wall_clock_latency():
    fake = FakeRetrieveOnce()
    state = _state()
    request = RetrieveRequest(kb_id="kb-1", user_id="u1", query="原问题")
    started_at = perf_counter()
    result = asyncio.run(
        ResearchParallelRetrievalService(fake).execute(
            state=state,
            queries=_queries(),
            request=request,
            knowledge_bases=[],
            round=1,
        )
    )
    wall_ms = int((perf_counter() - started_at) * 1000)

    assert fake.max_active == 3
    assert result.success_count == 3
    assert result.failed_count == 0
    assert result.latency_ms < 80
    assert result.latency_ms <= wall_ms + 10
    assert all(call["tenant_id"] == "tenant-1" for call in fake.calls)
    assert all(call["index_versions"] == {"kb-1": "v2"} for call in fake.calls)
    assert all(call["request"] is request for call in fake.calls)
    assert state.visited_queries == ["查询1", "查询2", "查询3"]


def test_one_failed_task_does_not_cancel_other_queries():
    fake = FakeRetrieveOnce(failed_query_id="Q2")
    result = asyncio.run(
        ResearchParallelRetrievalService(fake).execute(
            state=_state(),
            queries=_queries(),
            request=RetrieveRequest(kb_id="kb-1", user_id="u1", query="原问题"),
            knowledge_bases=[],
            round=1,
        )
    )

    assert result.success_count == 2
    assert result.failed_count == 1
    assert result.degraded is True
    assert [task.success for task in result.tasks] == [True, False, True]
    assert result.tasks[1].query_id == "Q2"
    assert result.tasks[1].error == "RuntimeError"


def test_all_failed_tasks_mark_round_degraded():
    class AlwaysFail:
        async def execute(self, **kwargs):
            raise RuntimeError("no backend")

    result = asyncio.run(
        ResearchParallelRetrievalService(AlwaysFail()).execute(
            state=_state(),
            queries=_queries(),
            request=RetrieveRequest(kb_id="kb-1", user_id="u1", query="原问题"),
            knowledge_bases=[],
            round=1,
        )
    )
    assert result.success_count == 0
    assert result.failed_count == 3
    assert result.degraded is True
