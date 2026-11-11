from types import SimpleNamespace

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.schemas.rag import EvidenceItem, EvidencePack
from app.schemas.research import ResearchAspect, ResearchRequest
from app.services.research_decider_service import DeciderResult
from app.services.research_finalizer_service import FinalizerResult
from app.services.research_planner_service import PlannerResult
from app.services.research_service import ResearchRunResult, ResearchService
from app.tenant.plan_presets import PLAN_PRESETS


def test_only_pro_plan_enables_research():
    assert PLAN_PRESETS["free"].features.research_allowed is False
    assert PLAN_PRESETS["standard"].features.research_allowed is False
    assert PLAN_PRESETS["pro"].features.research_allowed is True


@pytest.mark.parametrize("plan", ["free", "standard"])
def test_non_pro_research_is_rejected_before_loading_knowledge_bases(monkeypatch, plan):
    class FakeTenantRepository:
        def __init__(self, db):
            pass

        def get_by_id(self, tenant_id):
            return SimpleNamespace(plan=plan)

    class FakeRag:
        db = SimpleNamespace()
        chunk_repository = SimpleNamespace()
        retrieve_once_service = SimpleNamespace()

        def _load_knowledge_bases(self, *args, **kwargs):
            raise AssertionError("knowledge bases must not be loaded for a forbidden plan")

    monkeypatch.setattr("app.services.research_service.TenantRepository", FakeTenantRepository)
    service = ResearchService(rag_service=FakeRag())
    with pytest.raises(AppError) as exc_info:
        import asyncio

        asyncio.run(
            service.research(
                "tenant",
                ResearchRequest(kb_id="kb", user_id="u", query="问题"),
            )
        )
    assert exc_info.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code


def test_pro_research_checks_rate_once_without_local_retrieval_log_write(monkeypatch):
    class FakeTenantRepository:
        def __init__(self, db):
            pass

        def get_by_id(self, tenant_id):
            return SimpleNamespace(plan="pro")

    class FakeRateLimit:
        def __init__(self):
            self.checks = 0
            self.successes = 0

        def check_retrieve(self, *args, **kwargs):
            self.checks += 1

        def record_retrieve_success(self, tenant_id):
            self.successes += 1

    class FakeDb:
        def __init__(self):
            self.commits = 0

        def commit(self):
            self.commits += 1

    class FakeObservability:
        def __init__(self, **kwargs):
            self.trace_id = "trace-1"

        def start(self, **kwargs):
            pass

        def record_planner(self, value):
            pass

        def record_round(self, value):
            pass

        def record_decider(self, value, **kwargs):
            pass

        def record_finalizer(self, value):
            pass

        def finish(self, **kwargs):
            self.stop_reason = kwargs["stop_reason"]

    db = FakeDb()
    rag = SimpleNamespace(
        db=db,
        chunk_repository=SimpleNamespace(),
        retrieve_once_service=SimpleNamespace(),
        vector_store=SimpleNamespace(name="pgvector"),
        _load_knowledge_bases=lambda tenant_id, kb_ids: [
            SimpleNamespace(id="kb", name="规则库", description="规则", active_index_version="v2")
        ],
        _resolve_index_versions=lambda *args, **kwargs: {"kb": "v2"},
    )
    rate_limit = FakeRateLimit()
    monkeypatch.setattr("app.services.research_service.TenantRepository", FakeTenantRepository)
    monkeypatch.setattr(
        "app.services.research_service.get_rate_limit_service",
        lambda: rate_limit,
    )
    monkeypatch.setattr(
        "app.services.research_service.ResearchObservability",
        FakeObservability,
    )
    service = ResearchService(rag_service=rag)

    evidence_pack = EvidencePack(
        status="complete",
        groups=[],
        items=[
            EvidenceItem(
                evidence_id="E1",
                chunk_id="chunk-1",
                kb_id="kb",
                document_id="doc-1",
                title="规则",
                index_version="v2",
                content="真实证据",
                source="hybrid",
            )
        ],
    )

    async def fake_run(*, state, request, knowledge_bases, allow_supplement=True):
        assert allow_supplement is True
        assert request.profile == "quality"
        assert request.index_version == "v2"
        assert request.top_k == 10
        assert request.retrieval_options.mode == "hybrid"
        assert request.retrieval_options.vector_top_k == 20
        assert request.retrieval_options.bm25_top_k == 20
        assert request.retrieval_options.rrf_k == 10
        assert request.rerank_options.enabled is True
        assert request.rerank_options.top_n == 10
        assert request.query_options.enabled is False
        assert request.query_options.synonym_enabled is False
        assert request.evidence_options.enabled is False
        aspect = ResearchAspect(aspect_id="A1", description="规则")
        state.aspects = [aspect]
        state.stop_reason = "evidence_complete"
        state.round_count = 1
        state.retrieval_task_count = 1
        state.llm_call_count = 3
        state.input_tokens = 30
        state.output_tokens = 10
        state.elapsed_ms = 12
        planner = PlannerResult(
            aspects=[aspect],
            queries=[],
            raw_output={},
            model="fast",
            latency_ms=1,
        )
        decider = DeciderResult(
            pack=evidence_pack,
            selections={},
            next_queries=[],
            missing_aspect_ids=[],
            raw_output={},
            model="fast",
            latency_ms=1,
        )
        finalizer = FinalizerResult(
            answer="回答[E1]",
            pack=evidence_pack,
            raw_output={},
            model="strong",
            latency_ms=1,
        )
        return ResearchRunResult(
            state=state,
            answer="回答[E1]",
            evidence_pack=evidence_pack,
            candidates=[],
            planner=planner,
            decider=decider,
            finalizer=finalizer,
        )

    service.run = fake_run
    import asyncio

    data = asyncio.run(
        service.research(
            "tenant",
            ResearchRequest(kb_id="kb", user_id="u", query="问题"),
        )
    )
    assert rate_limit.checks == 1
    assert rate_limit.successes == 1
    assert db.commits == 0
    assert data.metadata.stop_reason == "evidence_complete"
    assert data.metadata.profile == "research_fixed"
    assert data.metadata.trace_id == "trace-1"
