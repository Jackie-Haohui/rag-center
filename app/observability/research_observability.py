from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.observability.langfuse_client import get_langfuse_client
from app.schemas.research import ResearchRound
from app.services.research_decider_service import DeciderResult
from app.services.research_finalizer_service import FinalizerResult
from app.services.research_planner_service import PlannerResult

logger = get_logger(__name__)


class ResearchObservability:
    """Best-effort Research tracing that never changes the business result."""

    def __init__(self, *, research_id: str, log_id: str) -> None:
        self.research_id = research_id
        self.log_id = log_id
        self.trace_id: str | None = None
        self._trace: Any | None = None
        self._langfuse: Any | None = None

    def start(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        kb_ids: list[str],
        plan: str,
        profile: str,
        index_versions: dict[str, str],
    ) -> None:
        try:
            langfuse = get_langfuse_client()
            if langfuse is None:
                return
            self._langfuse = langfuse
            self._trace = langfuse.trace(
                name="rag_research",
                user_id=user_id,
                input={"query": query, "kb_ids": kb_ids, "profile": profile},
                metadata={
                    "research_id": self.research_id,
                    "log_id": self.log_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "plan": plan,
                    "profile": profile,
                    "index_versions": index_versions,
                },
            )
            self.trace_id = self._trace.id
        except Exception as exc:  # noqa: BLE001 - telemetry is isolated from the API.
            self._warn("start", exc)

    def record_planner(self, result: PlannerResult) -> None:
        self._span(
            "planner",
            {
                "model": result.model,
                "role": "fast",
                "aspect_count": len(result.aspects),
                "query_count": len(result.queries),
                "latency_ms": result.latency_ms,
                **self._usage(result.usage),
                "degraded": result.degraded,
            },
        )

    def record_round(self, value: ResearchRound) -> None:
        self._span(
            f"retrieval_round_{value.round}",
            {
                "round": value.round,
                "purpose": value.purpose,
                "candidate_count": value.candidate_count,
                "new_chunk_count": value.new_chunk_count,
                "latency_ms": value.latency_ms,
                "tasks": [
                    {
                        "query_id": task.query_id,
                        "query": task.query,
                        "aspect_ids": task.aspect_ids,
                        "success": task.success,
                        "chunk_ids": task.chunk_ids,
                        "chunk_count": len(task.chunk_ids),
                        "new_chunk_count": task.new_chunk_count,
                        "latency_ms": task.latency_ms,
                        "degraded": task.degraded,
                    }
                    for task in value.tasks
                ],
            },
        )

    def record_decider(self, result: DeciderResult, *, candidate_count: int) -> None:
        self._span(
            "evidence_decider",
            {
                "model": result.model,
                "role": "fast",
                "candidate_count": candidate_count,
                "covered_aspect_count": sum(group.covered for group in result.pack.groups),
                "missing_aspect_count": len(result.missing_aspect_ids),
                "next_query_count": len(result.next_queries),
                "latency_ms": result.latency_ms,
                **self._usage(result.usage),
                "degraded": result.degraded,
            },
        )

    def record_finalizer(self, result: FinalizerResult | None) -> None:
        if result is None:
            return
        self._span(
            "finalizer",
            {
                "model": result.model,
                "role": "strong",
                "evidence_count": len(result.pack.items),
                "answer_chars": len(result.answer or ""),
                "status": result.pack.status,
                "latency_ms": result.latency_ms,
                **self._usage(result.usage),
                "degraded": result.degraded,
            },
        )

    def finish(
        self,
        *,
        status: str,
        evidence_chunk_ids: list[str],
        evidence_count: int,
        round_count: int,
        retrieval_task_count: int,
        llm_call_count: int,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str,
        latency_ms: int,
        degraded: bool,
    ) -> None:
        try:
            if self._trace is not None:
                self._trace.update(
                    output={
                        "evidence_status": status,
                        "evidence_count": evidence_count,
                        "evidence_chunk_ids": evidence_chunk_ids,
                        "round_count": round_count,
                        "retrieval_task_count": retrieval_task_count,
                        "llm_call_count": llm_call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "stop_reason": stop_reason,
                        "latency_ms": latency_ms,
                        "degraded": degraded,
                    }
                )
            if self._langfuse is not None:
                self._langfuse.flush()
        except Exception as exc:  # noqa: BLE001 - telemetry is isolated from the API.
            self._warn("finish", exc)

    def _span(self, name: str, output: dict[str, Any]) -> None:
        if self._trace is None:
            return
        try:
            self._trace.span(
                name=name,
                input={"stage": name},
                output=output,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry is isolated from the API.
            self._warn(name, exc)

    @staticmethod
    def _usage(usage: dict | None) -> dict[str, int]:
        value = usage or {}
        return {
            "input_tokens": int(value.get("prompt_tokens", 0) or 0),
            "output_tokens": int(value.get("completion_tokens", 0) or 0),
        }

    def _warn(self, stage: str, exc: Exception) -> None:
        logger.warning(
            "LANGFUSE_RESEARCH_FAILED | research_id=%s | log_id=%s | stage=%s | "
            "error_type=%s",
            self.research_id,
            self.log_id,
            stage,
            type(exc).__name__,
        )
