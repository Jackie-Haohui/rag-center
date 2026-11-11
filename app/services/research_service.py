from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ErrorCode
from app.core.http_clients import ExternalHttpClients
from app.observability.research_observability import ResearchObservability
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.tenant_repository import TenantRepository
from app.schemas.rag import (
    EvidenceOptions,
    EvidencePack,
    QueryOptions,
    RerankOptions,
    RetrievalOptions,
    RetrieveRequest,
)
from app.schemas.research import (
    ResearchCandidate,
    ResearchData,
    ResearchMetadata,
    ResearchPlanData,
    ResearchRequest,
    ResearchRound,
    ResearchRoundMetadata,
    ResearchStageUsage,
    ResearchState,
    ResearchTaskMetadata,
)
from app.services.evidence_pack_builder import EvidencePackBuilder
from app.services.rag_service import RagService
from app.services.rate_limit_service import get_rate_limit_service
from app.services.research_candidate_fusion_service import (
    CandidateFusionResult,
    ResearchCandidateFusionService,
)
from app.services.research_decider_service import DeciderResult, ResearchDeciderService
from app.services.research_finalizer_service import (
    FinalizerResult,
    ResearchFinalizerService,
)
from app.services.research_limits import (
    DEFAULT_RESEARCH_TIMEOUT_SECONDS,
    MAX_RESEARCH_LLM_CALLS,
    MAX_RESEARCH_ROUNDS,
)
from app.services.research_parallel_retrieval_service import (
    ParallelRetrievalResult,
    ResearchParallelRetrievalService,
)
from app.services.research_planner_service import PlannerResult, ResearchPlannerService
from app.services.research_supplement_service import (
    ResearchSupplementService,
    SupplementResult,
)
from app.tenant.plan_resolver import PlanResolver
from app.tenant.retrieve_policy import resolve_retrieve_request
from app.utils.id_generator import generate_uuid

RESEARCH_PROFILE = "quality"
RESEARCH_METADATA_PROFILE = "research_fixed"
RESEARCH_INDEX_VERSION = "v2"
RESEARCH_TOP_K = 10
RESEARCH_CANDIDATE_TOP_K = 20
RESEARCH_RRF_K = 10


@dataclass(slots=True)
class ResearchRunResult:
    state: ResearchState
    answer: str | None
    evidence_pack: EvidencePack
    candidates: list[ResearchCandidate]
    planner: PlannerResult
    decider: DeciderResult
    finalizer: FinalizerResult | None


class ResearchService:
    """Executes the bounded Planner → Retrieve → Decider → Supplement → Finalizer flow."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        rag_service: RagService | None = None,
        planner: ResearchPlannerService | None = None,
        decider: ResearchDeciderService | None = None,
        finalizer: ResearchFinalizerService | None = None,
        parallel_retrieval: ResearchParallelRetrievalService | None = None,
        fusion: ResearchCandidateFusionService | None = None,
        supplement: ResearchSupplementService | None = None,
        http_clients: ExternalHttpClients | None = None,
    ) -> None:
        if rag_service is None:
            if db is None:
                raise ValueError("db or rag_service is required")
            rag_service = RagService(db, http_clients=http_clients)
        self.rag_service = rag_service
        repository = getattr(rag_service, "chunk_repository", None)
        if repository is None:
            if db is None:
                raise ValueError("a chunk repository is required")
            repository = ChunkRepository(db)
        pack_builder = EvidencePackBuilder(repository)
        model_client = http_clients.model if http_clients else None
        self.planner = planner or ResearchPlannerService(
            LLMRoleRouter.provider_for(LLMRole.FAST, client=model_client)
        )
        self.parallel_retrieval = parallel_retrieval or ResearchParallelRetrievalService(
            rag_service.retrieve_once_service
        )
        self.fusion = fusion or ResearchCandidateFusionService()
        self.decider = decider or ResearchDeciderService(
            pack_builder,
            LLMRoleRouter.provider_for(LLMRole.FAST, client=model_client),
        )
        self.supplement = supplement or ResearchSupplementService(self.parallel_retrieval)
        self.finalizer = finalizer or ResearchFinalizerService(
            pack_builder,
            LLMRoleRouter.provider_for(LLMRole.STRONG, client=model_client),
        )

    async def research(
        self,
        tenant_id: str,
        request: ResearchRequest,
        *,
        allow_supplement: bool = True,
    ) -> ResearchData:
        tenant = TenantRepository(self.rag_service.db).get_by_id(tenant_id)
        if tenant is None:
            raise AppError(
                ErrorCode.NOT_FOUND,
                msg="租户不存在",
                context={"tenant_id": tenant_id},
            )
        plan_context = PlanResolver.resolve_plan(tenant)
        if not plan_context.features.research_allowed:
            raise AppError(
                ErrorCode.FEATURE_NOT_ALLOWED,
                msg=f"当前套餐（{plan_context.plan}）不支持 Research",
                context={"plan": plan_context.plan},
            )

        fixed_retrieval_options = RetrievalOptions(
            mode="hybrid",
            vector_top_k=RESEARCH_CANDIDATE_TOP_K,
            bm25_top_k=RESEARCH_CANDIDATE_TOP_K,
            rrf_k=RESEARCH_RRF_K,
        )
        fixed_rerank_options = RerankOptions(enabled=True, top_n=RESEARCH_TOP_K)
        fixed_evidence_options = EvidenceOptions(enabled=False)
        fixed_query_options = QueryOptions(
            enabled=False,
            strategy="noop",
            synonym_enabled=False,
        )
        retrieve_request = RetrieveRequest(
            **request.model_dump(),
            profile=RESEARCH_PROFILE,
            index_version=RESEARCH_INDEX_VERSION,
        )
        resolved_request, tenant_policy = resolve_retrieve_request(
            retrieve_request,
            plan_context,
        )
        # The ordinary retrieve policy honors deployment-wide defaults. Research
        # has a product-level fixed contract, so reapply it after permission and
        # knowledge-base-limit validation instead of silently inheriting a profile
        # or feature-switch variation.
        resolved_request = resolved_request.model_copy(
            update={
                "index_version": RESEARCH_INDEX_VERSION,
                "profile": RESEARCH_PROFILE,
                "top_k": RESEARCH_TOP_K,
                "retrieval_options": fixed_retrieval_options,
                "rerank_options": fixed_rerank_options,
                "evidence_options": fixed_evidence_options,
                "query_options": fixed_query_options,
            }
        )
        tenant_policy = tenant_policy.model_copy(
            update={
                "retrieve_profile": RESEARCH_METADATA_PROFILE,
                "effective_mode": "hybrid",
                "effective_rerank": True,
                "effective_evidence": False,
                "effective_query_rewrite": False,
            }
        )
        kb_ids = resolved_request.kb_ids or []
        knowledge_bases = self.rag_service._load_knowledge_bases(tenant_id, kb_ids)
        index_versions = self.rag_service._resolve_index_versions(
            tenant_id,
            knowledge_bases,
            requested_version=resolved_request.index_version,
            index_version_overrides=None,
        )

        get_rate_limit_service().check_retrieve(
            tenant_id,
            qps_limit=plan_context.limits.retrieve_qps,
            daily_limit=plan_context.limits.daily_retrieve_limit,
        )
        research_id = generate_uuid()
        log_id = generate_uuid()
        observability = ResearchObservability(research_id=research_id, log_id=log_id)
        observability.start(
            tenant_id=tenant_id,
            user_id=resolved_request.user_id,
            query=resolved_request.query,
            kb_ids=kb_ids,
            plan=plan_context.plan,
            profile=tenant_policy.retrieve_profile,
            index_versions=index_versions,
        )
        state = ResearchState(
            research_id=research_id,
            original_query=resolved_request.query,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            index_versions=index_versions,
        )
        run = await self.run(
            state=state,
            request=resolved_request,
            knowledge_bases=knowledge_bases,
            allow_supplement=allow_supplement,
        )
        stop_reason = run.state.stop_reason or "degraded"
        if run.state.stop_reason is None:
            run.state.stop_reason = stop_reason
            self._mark_degraded(run.state, "research stopped without an explicit reason")

        observability.record_planner(run.planner)
        if run.state.rounds:
            observability.record_round(run.state.rounds[0])
        observability.record_decider(run.decider, candidate_count=len(run.candidates))
        if len(run.state.rounds) > 1:
            observability.record_round(run.state.rounds[1])
        observability.record_finalizer(run.finalizer)
        observability.finish(
            status=run.evidence_pack.status,
            evidence_chunk_ids=[item.chunk_id for item in run.evidence_pack.items],
            evidence_count=len(run.evidence_pack.items),
            round_count=run.state.round_count,
            retrieval_task_count=run.state.retrieval_task_count,
            llm_call_count=run.state.llm_call_count,
            input_tokens=run.state.input_tokens,
            output_tokens=run.state.output_tokens,
            stop_reason=stop_reason,
            latency_ms=run.state.elapsed_ms,
            degraded=run.state.degraded,
        )

        get_rate_limit_service().record_retrieve_success(tenant_id)

        stages = [
            self._stage_usage(
                stage="planner",
                role="fast",
                model=run.planner.model,
                latency_ms=run.planner.latency_ms,
                usage=run.planner.usage,
                degraded=run.planner.degraded,
            ),
            self._stage_usage(
                stage="decider",
                role="fast",
                model=run.decider.model,
                latency_ms=run.decider.latency_ms,
                usage=run.decider.usage,
                degraded=run.decider.degraded,
            ),
        ]
        if run.finalizer is not None:
            stages.append(
                self._stage_usage(
                    stage="finalizer",
                    role="strong",
                    model=run.finalizer.model,
                    latency_ms=run.finalizer.latency_ms,
                    usage=run.finalizer.usage,
                    degraded=run.finalizer.degraded,
                )
            )
        return ResearchData(
            query=resolved_request.query,
            kb_id=kb_ids[0],
            kb_ids=kb_ids,
            answer=run.answer,
            evidence_pack=run.evidence_pack,
            plan=ResearchPlanData(
                problem_type=run.planner.problem_type,
                aspects=run.planner.aspects,
                initial_queries=run.planner.queries,
                degraded=run.planner.degraded,
                error=run.planner.error,
            ),
            rounds=[self._round_metadata(item) for item in run.state.rounds],
            metadata=ResearchMetadata(
                research_id=research_id,
                log_id=log_id,
                trace_id=observability.trace_id,
                profile=tenant_policy.retrieve_profile,
                plan=plan_context.plan,
                index_versions=index_versions,
                round_count=run.state.round_count,
                retrieval_task_count=run.state.retrieval_task_count,
                llm_call_count=run.state.llm_call_count,
                input_tokens=run.state.input_tokens,
                output_tokens=run.state.output_tokens,
                latency_ms=run.state.elapsed_ms,
                initial_evidence_status=run.decider.pack.status,
                stop_reason=stop_reason,
                degraded=run.state.degraded,
                errors=run.state.errors,
                stages=stages,
            ),
        )

    async def run(
        self,
        *,
        state: ResearchState,
        request: RetrieveRequest,
        knowledge_bases: list[Any],
        allow_supplement: bool = True,
    ) -> ResearchRunResult:
        started_at = perf_counter()

        state.stage = "planning"
        self._reserve_llm_call(state)
        planner_result = await self.planner.plan(
            question=state.original_query,
            knowledge_bases=knowledge_bases,
        )
        state.record_usage(planner_result.usage)
        state.problem_type = planner_result.problem_type
        state.aspects = planner_result.aspects
        state.initial_queries = planner_result.queries
        if planner_result.degraded:
            self._mark_degraded(state, planner_result.error)

        state.stage = "retrieving"
        round_one = await self.parallel_retrieval.execute(
            state=state,
            queries=planner_result.queries,
            request=request,
            knowledge_bases=knowledge_bases,
            round=1,
        )
        if round_one.success_count == 0:
            raise AppError(
                ErrorCode.RETRIEVAL_FAILED,
                internal_msg="all initial Research retrieval tasks failed",
                context={"research_id": state.research_id},
            )
        first_fusion = self.fusion.fuse(
            results=round_one.results,
            aspects=state.aspects,
        )
        self._record_round(
            state,
            retrieval=round_one,
            fusion=first_fusion,
            purpose="initial",
            new_chunk_count=len(first_fusion.candidates),
        )
        if round_one.degraded:
            self._mark_degraded(state, "initial retrieval degraded")

        state.stage = "deciding"
        self._reserve_llm_call(state)
        decider_result = await self.decider.decide(
            state=state,
            candidates=first_fusion.candidates,
        )
        state.record_usage(decider_result.usage)
        state.missing_aspect_ids = decider_result.missing_aspect_ids
        if decider_result.degraded:
            self._mark_degraded(state, decider_result.error)

        all_results = list(round_one.results)
        final_fusion = first_fusion
        if decider_result.pack.status == "complete":
            state.stop_reason = "evidence_complete"
        elif not allow_supplement:
            # This switch is intentionally service-internal. It exists only so the
            # offline A/B/C runner can isolate decomposition from second-round repair.
            state.stop_reason = "max_rounds"
        elif decider_result.degraded:
            state.stop_reason = "degraded"
        elif decider_result.all_next_queries_duplicate:
            state.stop_reason = "duplicate_query"
        elif not decider_result.next_queries:
            state.stop_reason = "degraded"
            self._mark_degraded(state, "no valid supplement query for missing evidence")
        elif self._timed_out(started_at):
            state.stop_reason = "timeout"
            self._mark_degraded(state, "research timeout before supplement retrieval")
        elif state.round_count >= MAX_RESEARCH_ROUNDS:
            state.stop_reason = "max_rounds"
        else:
            state.stage = "supplementing"
            supplement_result = await self.supplement.execute(
                state=state,
                queries=decider_result.next_queries,
                request=request,
                knowledge_bases=knowledge_bases,
                existing_candidates=first_fusion.candidates,
            )
            all_results.extend(supplement_result.retrieval.results)
            supplement_fusion = self.fusion.fuse(
                results=supplement_result.retrieval.results,
                aspects=state.aspects,
            )
            final_fusion = self.fusion.fuse(
                results=all_results,
                aspects=state.aspects,
            )
            self._record_supplement_round(
                state,
                supplement=supplement_result,
                fusion=supplement_fusion,
            )
            if supplement_result.retrieval.degraded:
                self._mark_degraded(state, "supplement retrieval degraded")
            if supplement_result.retrieval.success_count == 0:
                state.stop_reason = "degraded"
            elif supplement_result.new_chunk_count == 0:
                state.stop_reason = "no_new_evidence"
            else:
                state.stop_reason = "max_rounds"

        fallback_pack = decider_result.pack
        finalizer_result: FinalizerResult | None = None
        answer: str | None = None
        pack = fallback_pack
        if self._timed_out(started_at):
            state.stop_reason = "timeout"
            self._mark_degraded(state, "research timeout before finalizer")
        elif state.llm_call_count >= MAX_RESEARCH_LLM_CALLS:
            state.stop_reason = state.stop_reason or "budget_exhausted"
            self._mark_degraded(state, "research LLM budget exhausted before finalizer")
        else:
            state.stage = "finalizing"
            self._reserve_llm_call(state)
            finalizer_result = await self.finalizer.finalize(
                state=state,
                candidates=final_fusion.candidates,
                fallback_pack=fallback_pack,
            )
            state.record_usage(finalizer_result.usage)
            answer = finalizer_result.answer
            pack = finalizer_result.pack
            if finalizer_result.degraded:
                self._mark_degraded(state, finalizer_result.error)

        state.stage = "completed"
        state.elapsed_ms = int((perf_counter() - started_at) * 1000)
        state.candidate_chunk_ids = [
            candidate.chunk_id for candidate in final_fusion.candidates if candidate.chunk_id
        ]
        return ResearchRunResult(
            state=state,
            answer=answer,
            evidence_pack=pack,
            candidates=final_fusion.candidates,
            planner=planner_result,
            decider=decider_result,
            finalizer=finalizer_result,
        )

    @staticmethod
    def _record_round(
        state: ResearchState,
        *,
        retrieval: ParallelRetrievalResult,
        fusion: CandidateFusionResult,
        purpose: str,
        new_chunk_count: int,
    ) -> None:
        state.rounds.append(
            ResearchRound(
                round=len(state.rounds) + 1,
                purpose=purpose,
                tasks=retrieval.tasks,
                candidate_chunk_ids=[
                    candidate.chunk_id
                    for candidate in fusion.candidates
                    if candidate.chunk_id
                ],
                candidate_count=len(fusion.candidates),
                new_chunk_count=new_chunk_count,
                latency_ms=retrieval.latency_ms,
            )
        )
        state.round_count += 1
        state.retrieval_task_count += len(retrieval.tasks)

    def _record_supplement_round(
        self,
        state: ResearchState,
        *,
        supplement: SupplementResult,
        fusion: CandidateFusionResult,
    ) -> None:
        self._record_round(
            state,
            retrieval=supplement.retrieval,
            fusion=fusion,
            purpose="supplement",
            new_chunk_count=supplement.new_chunk_count,
        )

    @staticmethod
    def _reserve_llm_call(state: ResearchState) -> None:
        if state.llm_call_count >= MAX_RESEARCH_LLM_CALLS:
            state.stop_reason = "budget_exhausted"
            raise RuntimeError("research LLM call budget exhausted")
        state.llm_call_count += 1

    @staticmethod
    def _mark_degraded(state: ResearchState, error: str | None) -> None:
        state.degraded = True
        if error and error not in state.errors:
            state.errors.append(error[:300])

    @staticmethod
    def _timed_out(started_at: float) -> bool:
        return perf_counter() - started_at >= DEFAULT_RESEARCH_TIMEOUT_SECONDS

    @staticmethod
    def _round_metadata(value: ResearchRound) -> ResearchRoundMetadata:
        return ResearchRoundMetadata(
            round=value.round,
            purpose=value.purpose,
            tasks=[
                ResearchTaskMetadata(
                    query_id=task.query_id,
                    query=task.query,
                    aspect_ids=task.aspect_ids,
                    success=task.success,
                    chunk_count=len(task.chunk_ids),
                    chunk_ids=task.chunk_ids,
                    new_chunk_count=task.new_chunk_count,
                    latency_ms=task.latency_ms,
                    degraded=task.degraded,
                    error=task.error,
                )
                for task in value.tasks
            ],
            candidate_chunk_ids=value.candidate_chunk_ids,
            candidate_count=value.candidate_count,
            new_chunk_count=value.new_chunk_count,
            latency_ms=value.latency_ms,
        )

    @staticmethod
    def _stage_usage(
        *,
        stage: str,
        role: str,
        model: str,
        latency_ms: int,
        usage: dict | None,
        degraded: bool,
    ) -> ResearchStageUsage:
        value = usage or {}
        return ResearchStageUsage(
            stage=stage,
            role=role,
            model=model,
            latency_ms=latency_ms,
            input_tokens=int(value.get("prompt_tokens", 0) or 0),
            output_tokens=int(value.get("completion_tokens", 0) or 0),
            degraded=degraded,
        )
