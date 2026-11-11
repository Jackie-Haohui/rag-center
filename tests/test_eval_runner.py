import asyncio
from pathlib import Path

import pytest

from app.core.index_versions import LEGACY_INDEX_VERSION
from eval.core.dataset import load_dataset
from eval.core.experiment import load_experiment
from eval.core.models import ExperimentGroup
from eval.core.runner import (
    EvaluationRunError,
    _assert_index_version,
    _assert_legacy_index,
    _build_payload,
    _contexts_from_chunks,
    _experiment_snapshot,
    _retrieve_shared_candidates_bounded,
    _run_cases_bounded,
    _run_evidence_cases_bounded,
    _run_shared_rerank_cases_bounded,
    _success_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = load_dataset(PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json")
RERANK_EXPERIMENT = load_experiment(
    PROJECT_ROOT / "eval" / "experiments" / "rerank_upgrade.json",
    project_root=PROJECT_ROOT,
)


def _group() -> ExperimentGroup:
    return ExperimentGroup(
        label="test",
        profile="custom",
        top_k=5,
        retrieval_options={"mode": "vector"},
        rerank_options={"enabled": False},
        evidence_options={"enabled": False},
        query_options={
            "enabled": False,
            "strategy": "noop",
            "synonym_enabled": False,
        },
    )


def test_named_profile_snapshot_can_be_reloaded():
    experiment = load_experiment(
        PROJECT_ROOT / "eval" / "experiments" / "opt20_v2_speed_vs_balanced_regression.json",
        project_root=PROJECT_ROOT,
    )

    snapshot = _experiment_snapshot(experiment)

    assert snapshot["baseline"]["profile"] == "speed"
    assert snapshot["candidate"]["profile"] == "balanced"
    assert "top_k" not in snapshot["baseline"]
    assert "retrieval_options" not in snapshot["candidate"]
    assert type(experiment).model_validate(snapshot) == experiment


def test_payload_explicitly_locks_first_phase_index():
    case = DATASET.cases_for_suite("basic")[0]
    payload = _build_payload(
        case=case,
        kb_ids=[DATASET.knowledge_bases["ecommerce"].kb_id],
        group=_group(),
    )

    assert payload["index_version"] == LEGACY_INDEX_VERSION
    assert payload["observability_enabled"] is False
    assert payload["evidence_options"]["enabled"] is False


def test_non_first_phase_context_is_rejected():
    with pytest.raises(EvaluationRunError, match="错误索引版本"):
        _assert_legacy_index([{"index_version": "v2"}])


def test_payload_uses_group_index_version_and_context_expansion_text():
    case = DATASET.cases_for_suite("basic")[0]
    group = _group().model_copy(update={"index_version": "v2"})
    payload = _build_payload(
        case=case,
        kb_ids=[DATASET.knowledge_bases["ecommerce"].kb_id],
        group=group,
    )
    contexts = _contexts_from_chunks(
        [
            {
                "document_id": "doc",
                "chunk_id": "chunk",
                "title": "规则.md",
                "content": "原始命中",
                "index_version": "v2",
                "context": {"content": "原始命中\n\n补充证据"},
            }
        ]
    )

    assert payload["index_version"] == "v2"
    assert contexts[0]["text"] == "原始命中\n\n补充证据"
    assert contexts[0]["anchor_text"] == "原始命中"
    _assert_index_version(contexts, "v2")


def test_success_result_preserves_query_rewrite_details():
    case = DATASET.cases_for_suite("basic")[0]
    result = _success_result(
        case=case,
        group_name="candidate",
        kb_aliases=["ecommerce"],
        data={
            "metadata": {
                "query_processing": {
                    "enabled": True,
                    "strategy": "rewrite",
                    "raw_query": case.question,
                    "effective_query": "现货普通订单付款后揽收时限",
                    "search_query": "现货普通订单付款后揽收时限",
                    "latency_ms": 123,
                    "synonym_expansions": [],
                    "degraded": False,
                },
                "rerank": {"enabled": False},
                "evidence": {"executed": False},
            }
        },
        contexts=[],
        effective_config={},
        latency_ms=456,
    )

    assert result["query_processing"] == {
        "enabled": True,
        "strategy": "rewrite",
        "raw_query": case.question,
        "effective_query": "现货普通订单付款后揽收时限",
        "search_query": "现货普通订单付款后揽收时限",
        "synonym_expansions": [],
        "degraded": False,
        "degraded_reason": None,
    }


def test_bounded_runner_preserves_order_and_concurrency(monkeypatch):
    active = 0
    maximum = 0

    async def fake_run_case(*, case, keep_result, **kwargs):
        nonlocal active, maximum
        assert keep_result is True
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"case_id": case.id, "error": None}

    monkeypatch.setattr("eval.core.runner._run_case", fake_run_case)
    cases = DATASET.cases_for_suite("basic")[:6]
    results = asyncio.run(
        _run_cases_bounded(
            client=object(),
            dataset=DATASET,
            cases=cases,
            group_name="baseline",
            group=_group(),
            concurrency=2,
        )
    )

    assert maximum == 2
    assert [item["case_id"] for item in results] == [case.id for case in cases]


def test_rerank_experiment_reuses_one_shared_top20_for_both_groups():
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self):
            self.payloads = []

        async def post(self, path, *, json):
            assert path == "/api/v1/rag/retrieve"
            self.payloads.append(json)
            chunks = [
                {
                    "kb_id": json["kb_id"],
                    "document_id": f"doc-{index}",
                    "chunk_id": f"chunk-{index}",
                    "title": f"文档{index}.md",
                    "content": f"候选片段{index}",
                    "retrieval_text": f"候选片段{index}",
                    "score": 1 - index / 100,
                    "index_version": LEGACY_INDEX_VERSION,
                }
                for index in range(20)
            ]
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "retrieved_chunks": chunks,
                        "metadata": {
                            "top_k": 20,
                            "latency_ms": 50,
                            "tenant_policy": {
                                "retrieve_profile": "custom",
                                "effective_mode": "hybrid",
                                "effective_query_rewrite": False,
                            },
                            "query_processing": {"synonym_enabled": False},
                            "rerank": {"enabled": False},
                            "evidence": {"enabled": False},
                        },
                    },
                }
            )

    class FakeProvider:
        def __init__(self, reverse):
            self.reverse = reverse
            self.received = []

        async def rerank(self, *, query, chunks, top_n):
            self.received.append([chunk["chunk_id"] for chunk in chunks])
            ordered = list(reversed(chunks)) if self.reverse else chunks
            return [
                {**chunk, "rerank_score": 1 - index / 100}
                for index, chunk in enumerate(ordered[:top_n])
            ]

    cases = DATASET.cases_for_suite("basic")[:3]
    client = FakeClient()
    shared = asyncio.run(
        _retrieve_shared_candidates_bounded(
            client=client,
            dataset=DATASET,
            cases=cases,
            experiment=RERANK_EXPERIMENT,
        )
    )
    llm = FakeProvider(reverse=False)
    qwen3 = FakeProvider(reverse=True)
    baseline = asyncio.run(
        _run_shared_rerank_cases_bounded(
            shared_rows=shared,
            cases=cases,
            group_name="baseline",
            group=RERANK_EXPERIMENT.baseline,
            provider=llm,
            concurrency=2,
        )
    )
    candidate = asyncio.run(
        _run_shared_rerank_cases_bounded(
            shared_rows=shared,
            cases=cases,
            group_name="candidate",
            group=RERANK_EXPERIMENT.candidate,
            provider=qwen3,
            concurrency=2,
        )
    )

    assert len(client.payloads) == len(cases)
    assert all(payload["top_k"] == 20 for payload in client.payloads)
    assert all(payload["rerank_options"]["enabled"] is False for payload in client.payloads)
    assert all(payload["retrieval_options"]["mode"] == "hybrid" for payload in client.payloads)
    assert all(payload["query_options"]["enabled"] is False for payload in client.payloads)
    assert all(payload["query_options"]["synonym_enabled"] is False for payload in client.payloads)
    assert all("rerank_provider" not in payload for payload in client.payloads)
    assert llm.received == qwen3.received
    assert all(len(item["contexts"]) == 10 for item in baseline + candidate)
    assert all(item["application_model_calls"] == 1 for item in baseline + candidate)
    assert all(item["effective_config"]["index_version"] == "v1" for item in baseline + candidate)


def test_evidence_experiment_executes_once_and_splits_top10_from_pack():
    experiment = load_experiment(
        PROJECT_ROOT / "eval" / "experiments" / "evidence_orchestration.json",
        project_root=PROJECT_ROOT,
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            chunks = [
                {
                    "kb_id": DATASET.knowledge_bases["ecommerce"].kb_id,
                    "document_id": f"doc-{index}",
                    "chunk_id": f"chunk-{index}",
                    "title": f"文档{index}.md",
                    "content": f"原始片段{index}",
                    "index_version": "v2",
                }
                for index in range(10)
            ]
            return {
                "code": 0,
                "data": {
                    "retrieved_chunks": chunks,
                    "evidence_pack": {
                        "status": "complete",
                        "missing_aspects": [],
                        "items": [
                            {
                                "chunk_id": chunk["chunk_id"],
                                "kb_id": chunk["kb_id"],
                                "document_id": chunk["document_id"],
                                "title": chunk["title"],
                                "heading_path": "规则/退款",
                                "index_version": "v2",
                                "content": chunk["content"],
                                "source": "hybrid",
                            }
                            for chunk in chunks[:3]
                        ],
                    },
                    "metadata": {
                        "top_k": 10,
                        "latency_ms": 100,
                        "tenant_policy": {
                            "retrieve_profile": "custom",
                            "effective_mode": "hybrid",
                            "effective_query_rewrite": False,
                        },
                        "query_processing": {"synonym_enabled": False},
                        "rerank": {
                            "enabled": True,
                            "provider": "qwen3",
                            "latency_ms": 20,
                            "degraded": False,
                        },
                        "evidence": {
                            "enabled": True,
                            "executed": True,
                            "latency_ms": 10,
                            "degraded": False,
                        },
                    },
                },
            }

    class FakeClient:
        def __init__(self):
            self.payloads = []

        async def post(self, path, *, json):
            self.payloads.append(json)
            return FakeResponse()

    client = FakeClient()
    cases = DATASET.cases_for_suite("basic")[:2]
    groups, shared = asyncio.run(
        _run_evidence_cases_bounded(
            client=client,
            dataset=DATASET,
            cases=cases,
            experiment=experiment,
        )
    )

    assert len(client.payloads) == len(cases)
    assert all(
        payload["evidence_options"] == {"enabled": True, "max_items": 10}
        for payload in client.payloads
    )
    assert all(len(row["contexts"]) == 10 for row in groups["baseline"])
    assert all(len(row["contexts"]) == 3 for row in groups["candidate"])
    assert all(row["application_model_calls"] == 1 for row in groups["baseline"])
    assert all(row["application_model_calls"] == 2 for row in groups["candidate"])
    assert all(
        row["diagnostics"]["citation_traceability_rate"] == 1.0
        for row in groups["candidate"]
    )
    assert all(row["rerank_provider"] == "qwen3" for row in shared)
