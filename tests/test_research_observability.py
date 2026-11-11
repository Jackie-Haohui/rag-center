from types import SimpleNamespace

from app.observability import research_observability as module
from app.schemas.rag import EvidencePack
from app.schemas.research import (
    PlannedQuery,
    ResearchAspect,
    ResearchRound,
    ResearchTaskResult,
)
from app.services.research_decider_service import DeciderResult
from app.services.research_finalizer_service import FinalizerResult
from app.services.research_planner_service import PlannerResult


class FakeTrace:
    id = "trace-1"

    def __init__(self):
        self.spans = []
        self.output = None

    def span(self, **kwargs):
        self.spans.append(kwargs)

    def update(self, *, output):
        self.output = output


class FakeLangfuse:
    def __init__(self):
        self.trace_value = FakeTrace()
        self.trace_args = None
        self.flushed = False

    def trace(self, **kwargs):
        self.trace_args = kwargs
        return self.trace_value

    def flush(self):
        self.flushed = True


def test_research_trace_has_ordered_safe_spans_and_consistent_stop_reason(monkeypatch):
    langfuse = FakeLangfuse()
    monkeypatch.setattr(module, "get_langfuse_client", lambda: langfuse)
    value = module.ResearchObservability(research_id="research-1", log_id="log-1")
    value.start(
        tenant_id="tenant",
        user_id="user",
        query="问题",
        kb_ids=["kb"],
        plan="pro",
        profile="quality",
        index_versions={"kb": "v2"},
    )
    planner = PlannerResult(
        aspects=[ResearchAspect(aspect_id="A1", description="规则")],
        queries=[PlannedQuery(query_id="Q1", query="规则查询", aspect_ids=["A1"])],
        raw_output={"secret_candidate_body": "must not be traced"},
        model="fast",
        latency_ms=1,
    )
    round_one = ResearchRound(
        round=1,
        purpose="initial",
        tasks=[
            ResearchTaskResult(
                query_id="Q1",
                query="规则查询",
                aspect_ids=["A1"],
                round=1,
                success=True,
                latency_ms=2,
                chunk_ids=["chunk-1"],
            )
        ],
        candidate_count=1,
        new_chunk_count=1,
        latency_ms=2,
    )
    decider = DeciderResult(
        pack=EvidencePack(status="complete"),
        selections={},
        next_queries=[],
        missing_aspect_ids=[],
        raw_output={"candidate_body": "must not be traced"},
        model="fast",
        latency_ms=1,
    )
    finalizer = FinalizerResult(
        answer="完整回答不应写入 trace",
        pack=EvidencePack(status="complete"),
        raw_output={"answer": "must not be traced"},
        model="strong",
        latency_ms=1,
    )
    value.record_planner(planner)
    value.record_round(round_one)
    value.record_decider(decider, candidate_count=1)
    value.record_finalizer(finalizer)
    value.finish(
        status="complete",
        evidence_chunk_ids=["chunk-1"],
        evidence_count=1,
        round_count=1,
        retrieval_task_count=1,
        llm_call_count=3,
        input_tokens=10,
        output_tokens=5,
        stop_reason="evidence_complete",
        latency_ms=5,
        degraded=False,
    )

    assert langfuse.trace_args["name"] == "rag_research"
    assert [item["name"] for item in langfuse.trace_value.spans] == [
        "planner",
        "retrieval_round_1",
        "evidence_decider",
        "finalizer",
    ]
    serialized = str(langfuse.trace_value.spans)
    assert "must not be traced" not in serialized
    assert "完整回答" not in serialized
    assert langfuse.trace_value.output["stop_reason"] == "evidence_complete"
    assert langfuse.flushed is True


def test_langfuse_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        module,
        "get_langfuse_client",
        lambda: SimpleNamespace(trace=lambda **kwargs: (_ for _ in ()).throw(RuntimeError())),
    )
    value = module.ResearchObservability(research_id="r", log_id="l")
    value.start(
        tenant_id="t",
        user_id="u",
        query="q",
        kb_ids=["kb"],
        plan="pro",
        profile="quality",
        index_versions={"kb": "v2"},
    )
    assert value.trace_id is None
