import asyncio
from types import SimpleNamespace

from app.schemas.rag import RetrieveRequest
from app.schemas.research import PlannedQuery, ResearchCandidate, ResearchState
from app.services.research_supplement_service import ResearchSupplementService


class FakeParallel:
    def __init__(self, retrieval):
        self.retrieval = retrieval
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self.retrieval


def _state():
    return ResearchState(
        research_id="r1",
        original_query="问题",
        tenant_id="tenant",
        kb_ids=["kb"],
        index_versions={"kb": "v2"},
    )


def _candidate(chunk_id):
    return ResearchCandidate(
        chunk_id=chunk_id,
        kb_id="kb",
        document_id="doc",
        index_version="v2",
        title="规则",
        content="正文",
        source="hybrid",
    )


def test_supplement_caps_queries_and_counts_only_new_chunk_keys():
    retrieval = SimpleNamespace(
        results=[
            SimpleNamespace(
                retrieved_chunks=[
                    {"kb_id": "kb", "index_version": "v2", "chunk_id": "old"},
                    {"kb_id": "kb", "index_version": "v2", "chunk_id": "new"},
                ]
            )
        ]
    )
    parallel = FakeParallel(retrieval)
    service = ResearchSupplementService(parallel)
    queries = [
        PlannedQuery(query_id=f"Q{index}", query=f"查询{index}", aspect_ids=["A1"])
        for index in range(2, 5)
    ]
    result = asyncio.run(
        service.execute(
            state=_state(),
            queries=queries,
            request=RetrieveRequest(kb_id="kb", user_id="u", query="问题"),
            knowledge_bases=[],
            existing_candidates=[_candidate("old")],
        )
    )

    assert len(parallel.calls[0]["queries"]) == 2
    assert parallel.calls[0]["round"] == 2
    assert result.new_chunk_count == 1
