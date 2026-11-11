import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.rag import EvidencePack
from app.schemas.research import ResearchAspect, ResearchCandidate, ResearchState
from app.services.evidence_pack_builder import EvidencePackBuilder
from app.services.research_finalizer_service import ResearchFinalizerService


class FakeProvider:
    model = "deepseek-v4-pro"

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.last_usage = {"prompt_tokens": 30, "completion_tokens": 10}

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakeRepository:
    def __init__(self, chunks):
        self.chunks = {chunk.id: chunk for chunk in chunks}

    def get_by_ids_in_versioned_scope(self, chunk_ids, *, tenant_id, index_versions):
        return [self.chunks[item] for item in chunk_ids if item in self.chunks]


def _chunk(chunk_id, content):
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant",
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        title=f"规则-{chunk_id}",
        content=content,
        index_version="v2",
        metadata_={"heading_path": ["退款", chunk_id]},
    )


def _candidate(chunk_id, content):
    return ResearchCandidate(
        candidate_id="C1",
        chunk_id=chunk_id,
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        index_version="v2",
        title="模型标题",
        content="模型候选文本",
        source="hybrid",
        aspect_ids=["A1"],
        query_ids=["Q1"],
        rounds=[1],
        best_rank=1,
        context_relation="reference",
        metadata={"reference_texts": ["《促销活动与优惠券使用规则》第 4 节"]},
    )


def _state():
    return ResearchState(
        research_id="r1",
        original_query="退款规则",
        tenant_id="tenant",
        kb_ids=["kb"],
        index_versions={"kb": "v2"},
        aspects=[ResearchAspect(aspect_id="A1", description="退款")],
    )


def _run(output):
    chunks = [_chunk("c1", "数据库真实正文"), _chunk("c2", "另一条正文")]
    provider = FakeProvider(output)
    result = asyncio.run(
        ResearchFinalizerService(
            EvidencePackBuilder(FakeRepository(chunks)),
            provider,
        ).finalize(
            state=_state(),
            candidates=[_candidate("c1", "候选一"), _candidate("c2", "候选二")],
            fallback_pack=EvidencePack(status="missing"),
        )
    )
    return provider, result


def test_finalizer_uses_strong_model_and_backend_generates_stable_citations():
    provider, result = _run(
        {
            "aspects": [
                {"aspect_index": 0, "evidence": [{"candidate_index": 1, "role": "core"}]}
            ],
            "answer_blocks": [
                {"text": "退款按规则处理。", "evidence_indexes": [1], "kind": "supported"}
            ],
        }
    )
    assert result.answer == "退款按规则处理。[E1]"
    assert result.pack.status == "complete"
    assert result.pack.items[0].chunk_id == "c2"
    assert result.pack.items[0].content == "另一条正文"
    assert provider.calls[0]["thinking"] == "disabled"
    assert provider.calls[0]["max_tokens"] == 1800
    candidate_payload = provider.calls[0]["user_payload"]["candidates"][0]
    assert candidate_payload["context_relation"] == "reference"
    assert candidate_payload["reference_texts"] == ["《促销活动与优惠券使用规则》第 4 节"]


@pytest.mark.parametrize(
    "block",
    [
        {"text": "无引用结论", "evidence_indexes": [], "kind": "supported"},
        {"text": "带伪引用[E9]", "evidence_indexes": [0], "kind": "supported"},
        {"text": "越界", "evidence_indexes": [9], "kind": "supported"},
    ],
)
def test_invalid_answer_blocks_degrade_and_keep_fallback_pack(block):
    _, result = _run(
        {
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "answer_blocks": [block],
        }
    )
    assert result.answer is None
    assert result.pack.status == "missing"
    assert result.degraded is True


def test_insufficient_block_drops_echoed_evidence_without_degrading():
    _, result = _run(
        {
            "aspects": [{"aspect_index": 0, "evidence": []}],
            "answer_blocks": [
                {"text": "当前材料不足，无法确定。", "evidence_indexes": [0], "kind": "insufficient"}
            ],
        }
    )

    assert result.answer == "当前材料不足，无法确定。"
    assert result.pack.status == "missing"
    assert result.degraded is False


def test_all_insufficient_blocks_cannot_mark_evidence_complete():
    _, result = _run(
        {
            "aspects": [
                {
                    "aspect_index": 0,
                    "evidence": [{"candidate_index": 0, "role": "core"}],
                }
            ],
            "answer_blocks": [
                {
                    "text": "现有材料没有说明该场景，无法确定。",
                    "evidence_indexes": [],
                    "kind": "insufficient",
                }
            ],
        }
    )

    assert result.answer == "现有材料没有说明该场景，无法确定。"
    assert result.pack.status == "missing"
    assert result.pack.items == []
    assert result.degraded is False


def test_supported_label_cannot_turn_missing_mention_into_a_categorical_answer():
    _, result = _run(
        {
            "aspects": [
                {
                    "aspect_index": 0,
                    "evidence": [{"candidate_index": 0, "role": "core"}],
                }
            ],
            "answer_blocks": [
                {
                    "text": "规则未提及停车费，因此停车费不可以抵扣。",
                    "evidence_indexes": [0],
                    "kind": "supported",
                }
            ],
        }
    )

    assert result.answer == "当前材料不足，无法确定。"
    assert result.pack.status == "missing"
    assert result.pack.items == []
    assert result.degraded is False


def test_finalizer_failure_returns_decider_pack_without_inventing_answer():
    _, result = _run(RuntimeError("provider unavailable"))
    assert result.answer is None
    assert result.pack.status == "missing"
    assert result.error == "RuntimeError"
