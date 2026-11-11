import asyncio
from types import SimpleNamespace

from app.schemas.research import ResearchAspect, ResearchCandidate, ResearchState
from app.services.evidence_pack_builder import EvidencePackBuilder
from app.services.research_decider_service import ResearchDeciderService


class FakeProvider:
    model = "deepseek-v4-flash"

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.last_usage = {"prompt_tokens": 20, "completion_tokens": 8}

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


def _chunk(chunk_id):
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant",
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        title="规则",
        content=f"数据库正文-{chunk_id}",
        index_version="v2",
        metadata_={},
    )


def _candidate(chunk_id, aspects):
    return ResearchCandidate(
        candidate_id="C1",
        chunk_id=chunk_id,
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        index_version="v2",
        title="规则",
        content="模型输入正文",
        source="hybrid",
        aspect_ids=aspects,
        query_ids=["Q1"],
        rounds=[1],
        best_rank=1,
        context_relation="reference",
        metadata={"reference_texts": ["《仓储物流与发货履约 SLA》第 6 节"]},
    )


def _state():
    return ResearchState(
        research_id="r1",
        original_query="积分和现金怎么退",
        tenant_id="tenant",
        kb_ids=["kb"],
        index_versions={"kb": "v2"},
        aspects=[
            ResearchAspect(aspect_id="A1", description="积分"),
            ResearchAspect(aspect_id="A2", description="现金"),
        ],
        visited_queries=["积分和现金怎么退"],
    )


def _run(output, chunks=None, candidates=None):
    provider = FakeProvider(output)
    service = ResearchDeciderService(
        EvidencePackBuilder(FakeRepository(chunks or [_chunk("c1"), _chunk("c2")])),
        provider,
    )
    result = asyncio.run(
        service.decide(
            state=_state(),
            candidates=candidates
            or [_candidate("c1", ["A1"]), _candidate("c2", ["A2"])],
        )
    )
    return provider, result


def test_supporting_does_not_cover_aspect_and_next_query_targets_only_missing_aspect():
    provider, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]},
                {
                    "aspect_index": 1,
                    "evidence": [{"candidate_index": 1, "role": "supporting"}],
                },
            ],
            "decision": "stop",
            "next_queries": [{"query": "现金原路退款规则", "aspect_indexes": [1]}],
        }
    )

    assert result.pack.status == "partial"
    assert result.pack.missing_aspects == ["现金"]
    assert result.missing_aspect_ids == ["A2"]
    assert result.next_queries[0].aspect_ids == ["A2"]
    assert result.next_queries[0].query_id == "Q2"
    assert provider.calls[0]["thinking"] == "disabled"
    assert provider.calls[0]["max_tokens"] == 1200
    candidate_payload = provider.calls[0]["user_payload"]["candidates"][0]
    assert candidate_payload["context_relation"] == "reference"
    assert candidate_payload["reference_texts"] == ["《仓储物流与发货履约 SLA》第 6 节"]


def test_backend_ignores_supplement_suggestion_when_all_aspects_are_complete():
    _, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]},
                {"aspect_index": 1, "evidence": [{"candidate_index": 1, "role": "core"}]},
            ],
            "decision": "supplement",
            "next_queries": [{"query": "不应执行", "aspect_indexes": [0]}],
        }
    )
    assert result.pack.status == "complete"
    assert result.next_queries == []
    assert result.degraded is False


def test_duplicate_supplement_query_is_reported_without_execution():
    _, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": []},
                {"aspect_index": 1, "evidence": []},
            ],
            "decision": "supplement",
            "next_queries": [{"query": "积分和现金怎么退", "aspect_indexes": [0, 1]}],
        }
    )
    assert result.next_queries == []
    assert result.all_next_queries_duplicate is True
    assert result.degraded is False


def test_invalid_candidate_index_degrades_to_repository_safe_missing_pack():
    _, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 9, "role": "core"}]},
                {"aspect_index": 1, "evidence": []},
            ],
            "decision": "stop",
        }
    )
    assert result.degraded is True
    assert result.pack.status == "missing"
    assert result.pack.items == []


def test_repository_scope_mismatch_rejects_selected_model_candidate():
    wrong = _chunk("c1")
    wrong.tenant_id = "other"
    _, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core"}]},
                {"aspect_index": 1, "evidence": []},
            ],
            "decision": "stop",
        },
        chunks=[wrong, _chunk("c2")],
    )
    assert result.degraded is True
    assert result.pack.status == "missing"
