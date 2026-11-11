import asyncio
import inspect
from types import SimpleNamespace

from app.schemas.rag import (
    EvidencePack,
    RetrieveContextMetadata,
    RetrieveGraphInjectionMetadata,
    RetrieveRequest,
    RetrieveRerankMetadata,
)
from app.schemas.research import ResearchState
from app.services.evidence_pack_builder import EvidencePackBuilder
from app.services.research_decider_service import ResearchDeciderService
from app.services.research_finalizer_service import ResearchFinalizerService
from app.services.research_planner_service import ResearchPlannerService
from app.services.research_service import ResearchService
from app.services.retrieve_once_service import RetrieveOnceResult


class FakeProvider:
    def __init__(self, model, output):
        self.model = model
        self.output = output
        self.calls = []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakeRepository:
    def __init__(self, chunks):
        self.chunks = {chunk.id: chunk for chunk in chunks}

    def get_by_ids_in_versioned_scope(self, chunk_ids, *, tenant_id, index_versions):
        return [
            chunk
            for chunk_id in chunk_ids
            if (chunk := self.chunks.get(chunk_id))
            and chunk.tenant_id == tenant_id
            and index_versions.get(chunk.kb_id) == chunk.index_version
        ]


def _db_chunk(chunk_id, content):
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant",
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        title="规则",
        content=content,
        index_version="v2",
        metadata_={},
    )


def _retrieved_chunk(chunk_id, content):
    return {
        "chunk_id": chunk_id,
        "kb_id": "kb",
        "document_id": f"doc-{chunk_id}",
        "index_version": "v2",
        "title": "规则",
        "content": content,
        "retrieval_source": "hybrid",
        "metadata": {},
    }


class FakeRetrieveOnce:
    def __init__(self, by_query):
        self.by_query = by_query
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        chunk = self.by_query[kwargs["search_query"]]
        if isinstance(chunk, Exception):
            raise chunk
        return RetrieveOnceResult(
            query_id=kwargs["query_id"],
            query=kwargs["search_query"],
            aspect_ids=kwargs["aspect_ids"],
            round=kwargs["round"],
            candidate_snapshot=[chunk],
            retrieved_chunks=[chunk],
            retrieval_metadata={},
            candidate_filter_metadata={},
            graph_injection_metadata=RetrieveGraphInjectionMetadata(),
            context_metadata=RetrieveContextMetadata(),
            rerank_metadata=RetrieveRerankMetadata(enabled=False, provider="none"),
            latency_ms=2,
        )


def _state():
    return ResearchState(
        research_id="research-1",
        original_query="原问题",
        tenant_id="tenant",
        kb_ids=["kb"],
        index_versions={"kb": "v2"},
    )


def _build_service(*, decider_output, finalizer_output, by_query):
    chunks = [_db_chunk("c1", "首轮规则"), _db_chunk("c2", "补证规则")]
    repository = FakeRepository(chunks)
    retrieve_once = FakeRetrieveOnce(by_query)
    rag = SimpleNamespace(retrieve_once_service=retrieve_once, chunk_repository=repository)
    planner_provider = FakeProvider(
        "deepseek-v4-flash",
        {
            "aspects": ["必要规则"],
            "queries": [{"query": "初始查询", "aspect_indexes": [0]}],
        },
    )
    decider_provider = FakeProvider("deepseek-v4-flash", decider_output)
    finalizer_provider = FakeProvider("deepseek-v4-pro", finalizer_output)
    builder = EvidencePackBuilder(repository)
    service = ResearchService(
        rag_service=rag,
        planner=ResearchPlannerService(planner_provider),
        decider=ResearchDeciderService(builder, decider_provider),
        finalizer=ResearchFinalizerService(builder, finalizer_provider),
    )
    return service, retrieve_once, (planner_provider, decider_provider, finalizer_provider)


def test_complete_first_round_stops_without_supplement_and_uses_three_llm_calls():
    service, retrieve_once, providers = _build_service(
        by_query={"初始查询": _retrieved_chunk("c1", "首轮规则")},
        decider_output={
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]}
            ],
            "decision": "stop",
        },
        finalizer_output={
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]}
            ],
            "answer_blocks": [
                {"text": "按首轮规则处理。", "evidence_indexes": [0], "kind": "supported"}
            ],
        },
    )
    result = asyncio.run(
        service.run(
            state=_state(),
            request=RetrieveRequest(kb_id="kb", user_id="u", query="原问题"),
            knowledge_bases=[],
        )
    )

    assert result.state.stop_reason == "evidence_complete"
    assert result.state.round_count == 1
    assert result.state.retrieval_task_count == 1
    assert result.state.llm_call_count == 3
    assert result.answer == "按首轮规则处理。[E1]"
    assert len(retrieve_once.calls) == 1
    assert [len(provider.calls) for provider in providers] == [1, 1, 1]


def test_missing_evidence_runs_one_supplement_round_and_never_a_third_round():
    service, retrieve_once, _ = _build_service(
        by_query={
            "初始查询": _retrieved_chunk("c1", "首轮线索"),
            "定向补证": _retrieved_chunk("c2", "补证规则"),
        },
        decider_output={
            "aspects": [
                {
                    "aspect_index": 0,
                    "evidence": [{"candidate_index": 0, "role": "supporting"}],
                }
            ],
            "decision": "supplement",
            "next_queries": [{"query": "定向补证", "aspect_indexes": [0]}],
        },
        finalizer_output={
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 1, "role": "core"}]}
            ],
            "answer_blocks": [
                {"text": "补证后可确认。", "evidence_indexes": [1], "kind": "supported"}
            ],
        },
    )
    result = asyncio.run(
        service.run(
            state=_state(),
            request=RetrieveRequest(kb_id="kb", user_id="u", query="原问题"),
            knowledge_bases=[],
        )
    )

    assert result.state.stop_reason == "max_rounds"
    assert result.state.round_count == 2
    assert result.state.retrieval_task_count == 2
    assert result.state.llm_call_count == 3
    assert [call["round"] for call in retrieve_once.calls] == [1, 2]
    assert result.evidence_pack.status == "complete"
    assert result.answer == "补证后可确认。[E1]"
    round_metadata = ResearchService._round_metadata(result.state.rounds[1])
    assert round_metadata.tasks[0].chunk_ids == ["c2"]
    assert round_metadata.candidate_chunk_ids == ["c2"]


def test_offline_decomposition_mode_stops_after_first_round():
    service, retrieve_once, _ = _build_service(
        by_query={"初始查询": _retrieved_chunk("c1", "首轮线索")},
        decider_output={
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "decision": "supplement",
            "next_queries": [{"query": "定向补证", "aspect_indexes": [0]}],
        },
        finalizer_output={
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "answer_blocks": [
                {"text": "当前材料不足。", "evidence_indexes": [], "kind": "insufficient"}
            ],
        },
    )

    result = asyncio.run(
        service.run(
            state=_state(),
            request=RetrieveRequest(kb_id="kb", user_id="u", query="原问题"),
            knowledge_bases=[],
            allow_supplement=False,
        )
    )

    assert result.state.stop_reason == "max_rounds"
    assert result.state.round_count == 1
    assert result.state.retrieval_task_count == 1
    assert result.state.llm_call_count == 3
    assert len(retrieve_once.calls) == 1


def test_repeated_second_round_chunk_stops_with_no_new_evidence():
    service, _, _ = _build_service(
        by_query={
            "初始查询": _retrieved_chunk("c1", "首轮线索"),
            "定向补证": _retrieved_chunk("c1", "首轮线索"),
        },
        decider_output={
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "decision": "supplement",
            "next_queries": [{"query": "定向补证", "aspect_indexes": [0]}],
        },
        finalizer_output={
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "answer_blocks": [
                {"text": "当前材料不足。", "evidence_indexes": [], "kind": "insufficient"}
            ],
        },
    )
    result = asyncio.run(
        service.run(
            state=_state(),
            request=RetrieveRequest(kb_id="kb", user_id="u", query="原问题"),
            knowledge_bases=[],
        )
    )
    assert result.state.stop_reason == "no_new_evidence"
    assert result.state.round_count == 2


def test_finalizer_failure_keeps_decider_validated_pack():
    service, _, _ = _build_service(
        by_query={"初始查询": _retrieved_chunk("c1", "首轮规则")},
        decider_output={
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]}
            ],
            "decision": "stop",
        },
        finalizer_output=RuntimeError("unavailable"),
    )
    result = asyncio.run(
        service.run(
            state=_state(),
            request=RetrieveRequest(kb_id="kb", user_id="u", query="原问题"),
            knowledge_bases=[],
        )
    )
    assert result.answer is None
    assert result.evidence_pack.status == "complete"
    assert result.evidence_pack.items[0].content == "首轮规则"
    assert result.state.degraded is True
    assert result.state.llm_call_count == 3


def test_state_machine_has_no_unbounded_agent_loop():
    assert "while True" not in inspect.getsource(ResearchService.run)
    assert EvidencePack(status="missing").status == "missing"
