import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.schemas.rag import EvidenceOptions, RetrieveRequest
from app.services.evidence_orchestration_service import EvidenceOrchestrationService
from app.tenant.plan_presets import PLAN_PRESETS
from app.tenant.plan_resolver import PlanContext
from app.tenant.retrieve_policy import resolve_retrieve_request


class FakeLLM:
    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeRepository:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.calls: list[dict] = []

    def get_by_ids_in_versioned_scope(self, chunk_ids, **scope):
        self.calls.append({"chunk_ids": chunk_ids, **scope})
        return [
            self.chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.chunks
            and self.chunks[chunk_id].tenant_id == scope["tenant_id"]
            and scope["index_versions"].get(self.chunks[chunk_id].kb_id)
            == self.chunks[chunk_id].index_version
        ]


def chunk(chunk_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant",
        kb_id="kb",
        document_id=f"doc-{chunk_id}",
        title=f"title-{chunk_id}",
        content=content,
        retrieval_text=f"retrieval-{content}",
        index_version="v2",
        metadata_={"heading_path": f"规则/{chunk_id}"},
    )


def candidate(chunk_id: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "kb_id": "kb",
        "document_id": f"doc-{chunk_id}",
        "title": f"title-{chunk_id}",
        "content": content,
        "retrieval_text": f"retrieval-{content}",
        "index_version": "v2",
        "retrieval_source": "hybrid",
        "metadata": {"heading_path": f"规则/{chunk_id}"},
    }


def service(response: dict | Exception, chunks: list[SimpleNamespace]):
    llm = FakeLLM(response)
    value = EvidenceOrchestrationService(MagicMock(), llm_provider=llm)
    repository = FakeRepository(chunks)
    value.chunk_repository = repository
    return value, llm, repository


def run(service_value, *, candidates, retrieved, enabled=True, max_items=8):
    return asyncio.run(
        service_value.orchestrate(
            query="积分和现金分别怎么退？",
            tenant_id="tenant",
            kb_ids=["kb"],
            index_versions={"kb": "v2"},
            candidates=candidates,
            retrieved_chunks=retrieved,
            options=EvidenceOptions(enabled=enabled, max_items=max_items),
            rerank_enabled=False,
        )
    )


def test_disabled_evidence_does_not_call_llm():
    value, llm, _ = service({"aspects": []}, [])
    result = run(value, candidates=[], retrieved=[], enabled=False)
    assert result.pack is None
    assert result.metadata.enabled is False
    assert result.metadata.executed is False
    assert llm.calls == []


def test_candidate_pool_contains_topk_context_and_reserved_base_candidates():
    retrieved = [
        {
            **candidate("top", "top content"),
            "context": {
                "sources": [
                    {
                        "chunk_id": "top",
                        "relation": "anchor",
                        "content": "top content",
                    },
                    {
                        "chunk_id": "context",
                        "relation": "reference",
                        "content": "context content",
                        "title": "context",
                        "metadata": {},
                    },
                ]
            },
        }
    ]
    base = [candidate(f"base-{index}", f"base content {index}") for index in range(10)]
    built, included = EvidenceOrchestrationService.build_candidates(
        candidates=base,
        retrieved_chunks=retrieved,
    )
    ids = [item["chunk_id"] for item in built]
    assert ids[:2] == ["top", "context"]
    assert len([item for item in ids if item.startswith("base-")]) >= 6
    assert included is True


def test_pack_uses_original_chunks_and_allows_one_item_in_multiple_aspects():
    chunks = [chunk("top", "数据库中的原始正文"), chunk("outside", "现金原路退回")]
    response = {
        "aspects": [
            {"aspect": "积分", "evidence": [{"index": 0, "role": "core"}]},
            {
                "aspect": "现金",
                "evidence": [
                    {"index": 0, "role": "supporting"},
                    {"index": 1, "role": "core"},
                ],
            },
        ]
    }
    value, llm, repository = service(response, chunks)
    retrieved = [candidate("top", "模型输入正文")]
    result = run(
        value,
        candidates=[candidate("top", "模型输入正文"), candidate("outside", "现金原路退回")],
        retrieved=retrieved,
    )
    assert result.pack is not None
    assert result.pack.status == "complete"
    assert [item.chunk_id for item in result.pack.items] == ["top", "outside"]
    assert result.pack.items[0].content == "数据库中的原始正文"
    assert result.pack.groups[0].evidence[0].evidence_id == "E1"
    assert result.pack.groups[1].evidence[0].evidence_id == "E1"
    assert result.pack.items[1].retrieved_rank is None
    assert repository.calls[0]["tenant_id"] == "tenant"
    assert repository.calls[0]["index_versions"] == {"kb": "v2"}
    assert llm.calls[0]["log_payload"] is False


@pytest.mark.parametrize(
    "response",
    [
        {"aspects": [{"aspect": "退款", "evidence": [{"index": 9, "role": "core"}]}]},
        {"aspects": [{"aspect": "退款", "evidence": [{"index": 0, "role": "invalid"}]}]},
        {
            "aspects": [
                {
                    "aspect": "退款",
                    "evidence": [
                        {"index": 0, "role": "core"},
                        {"index": 0, "role": "supporting"},
                    ],
                }
            ]
        },
    ],
)
def test_invalid_model_output_degrades_without_partial_pack(response):
    value, _, _ = service(response, [chunk("top", "正文")])
    result = run(value, candidates=[candidate("top", "正文")], retrieved=[candidate("top", "正文")])
    assert result.pack is None
    assert result.metadata.degraded is True
    assert result.metadata.error


def test_repository_scope_mismatch_rejects_whole_pack():
    selected = chunk("top", "正文")
    selected.tenant_id = "other-tenant"
    value, _, _ = service(
        {"aspects": [{"aspect": "退款", "evidence": [{"index": 0, "role": "core"}]}]},
        [selected],
    )
    result = run(value, candidates=[candidate("top", "正文")], retrieved=[candidate("top", "正文")])
    assert result.pack is None
    assert result.metadata.degraded is True


def test_evidence_is_plan_limited(monkeypatch):
    monkeypatch.setattr(settings, "evidence_enabled", True)
    pro = PlanContext(
        plan="pro",
        features=PLAN_PRESETS["pro"].features,
        limits=PLAN_PRESETS["pro"].limits,
    )
    request = RetrieveRequest(
        kb_id="kb",
        user_id="user",
        query="test",
        profile="custom",
        evidence_options={"enabled": True, "max_items": 6},
    )
    resolved, policy = resolve_retrieve_request(request, pro)
    assert resolved.evidence_options.enabled is True
    assert policy.effective_evidence is True

    standard = PlanContext(
        plan="standard",
        features=PLAN_PRESETS["standard"].features,
        limits=PLAN_PRESETS["standard"].limits,
    )
    with pytest.raises(AppError) as exc_info:
        resolve_retrieve_request(request, standard)
    assert exc_info.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code
