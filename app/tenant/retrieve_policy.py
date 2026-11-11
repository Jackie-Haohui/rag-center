from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.core.logging import get_logger
from app.schemas.rag import (
    EvidenceOptions,
    QueryOptions,
    RerankOptions,
    RetrievalOptions,
    RetrieveRequest,
    TenantPolicyMetadata,
)
from app.tenant.plan_presets import DEFAULT_RETRIEVE_PROFILE, RetrieveProfileName
from app.tenant.plan_resolver import PlanContext
from app.tenant.retrieve_presets import FIXED_RRF_K, RETRIEVE_PROFILE_PRESETS

logger = get_logger(__name__)
NAMED_PROFILE_OVERRIDE_FIELDS = (
    "top_k",
    "retrieval_options",
    "rerank_options",
    "evidence_options",
    "query_options",
)


def resolve_retrieve_request(
    request: RetrieveRequest,
    plan_context: PlanContext,
) -> tuple[RetrieveRequest, TenantPolicyMetadata]:
    profile: RetrieveProfileName = request.profile or DEFAULT_RETRIEVE_PROFILE

    if profile not in plan_context.features.allowed_profiles:
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=f"当前套餐（{plan_context.plan}）不支持检索预设 {profile}",
            context={"plan": plan_context.plan, "profile": profile},
        )

    if profile == "custom":
        top_k = request.top_k if request.top_k is not None else settings.top_k
        retrieval_options = request.retrieval_options or RetrievalOptions(
            mode=settings.retrieval_mode,
        )
        if retrieval_options.rrf_k is not None:
            logger.warning(
                "RETRIEVE_RRF_K_DEPRECATED | requested_rrf_k=%s | effective_rrf_k=%s",
                retrieval_options.rrf_k,
                FIXED_RRF_K,
            )
        retrieval_options = retrieval_options.model_copy(
            update={"rrf_k": FIXED_RRF_K}
        )
        rerank_options = request.rerank_options or RerankOptions(
            enabled=settings.rerank_enabled,
        )
        evidence_options = request.evidence_options or EvidenceOptions(
            enabled=settings.evidence_enabled,
        )
        query_options = request.query_options or QueryOptions(
            enabled=settings.query_rewrite_enabled,
            strategy="rewrite" if settings.query_rewrite_enabled else "noop",
            synonym_enabled=False,
        )
    else:
        conflicting_fields = [
            field
            for field in NAMED_PROFILE_OVERRIDE_FIELDS
            if field in request.model_fields_set
        ]
        if conflicting_fields:
            raise AppError(
                ErrorCode.PARAM_ERROR,
                msg="命名检索档位不允许覆盖高级参数，请使用 custom",
                data={
                    "profile": profile,
                    "conflicting_fields": conflicting_fields,
                },
                context={
                    "profile": profile,
                    "conflicting_fields": conflicting_fields,
                },
            )
        preset = RETRIEVE_PROFILE_PRESETS[profile]
        assert preset is not None
        top_k = preset.top_k or settings.top_k
        retrieval_options = RetrievalOptions(**(preset.retrieval_options or {}))
        rerank_options = RerankOptions(**(preset.rerank_options or {}))
        evidence_options = EvidenceOptions(**(preset.evidence_options or {}))
        query_options = QueryOptions(**(preset.query_options or {}))

    _validate_plan_ceiling(
        plan_context,
        profile,
        retrieval_options,
        rerank_options,
        evidence_options,
        query_options,
    )

    kb_ids = request.kb_ids or []
    if len(kb_ids) > plan_context.limits.max_kb_per_retrieve:
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=(
                f"当前套餐（{plan_context.plan}）最多支持 "
                f"{plan_context.limits.max_kb_per_retrieve} 个知识库联合检索"
            ),
            context={
                "plan": plan_context.plan,
                "profile": profile,
                "kb_count": len(kb_ids),
                "max_kb_per_retrieve": plan_context.limits.max_kb_per_retrieve,
            },
        )

    requested_rerank = _requested_rerank_enabled(rerank_options)
    requested_evidence = _requested_evidence_enabled(evidence_options)
    requested_query_rewrite = _requested_query_rewrite_enabled(query_options)
    effective_rerank = requested_rerank and settings.rerank_enabled
    effective_evidence = requested_evidence and settings.evidence_enabled
    effective_query_rewrite = requested_query_rewrite and settings.query_rewrite_enabled

    rerank_options = RerankOptions(
        enabled=effective_rerank,
        top_n=rerank_options.top_n,
    )
    evidence_options = EvidenceOptions(
        enabled=effective_evidence,
        max_items=evidence_options.max_items,
    )
    if not effective_query_rewrite:
        query_options = QueryOptions(
            enabled=False,
            strategy="noop",
            synonym_enabled=query_options.synonym_enabled,
        )

    mode = retrieval_options.mode or settings.retrieval_mode
    resolved = request.model_copy(
        update={
            "top_k": top_k,
            "retrieval_options": retrieval_options,
            "rerank_options": rerank_options,
            "evidence_options": evidence_options,
            "query_options": query_options,
        }
    )
    tenant_policy = TenantPolicyMetadata(
        plan=plan_context.plan,
        retrieve_profile=profile,
        effective_mode=mode,
        effective_rerank=effective_rerank,
        effective_evidence=effective_evidence,
        effective_query_rewrite=effective_query_rewrite,
        rerank_disabled_reason=(
            "disabled_by_system" if requested_rerank and not effective_rerank else None
        ),
        evidence_disabled_reason=(
            "disabled_by_system" if requested_evidence and not effective_evidence else None
        ),
        query_rewrite_disabled_reason=(
            "disabled_by_system"
            if requested_query_rewrite and not effective_query_rewrite
            else None
        ),
        max_kb_per_retrieve=plan_context.limits.max_kb_per_retrieve,
        actual_kb_count=len(kb_ids),
    )
    return resolved, tenant_policy


def _validate_plan_ceiling(
    plan_context: PlanContext,
    profile: str,
    retrieval_options: RetrievalOptions,
    rerank_options: RerankOptions,
    evidence_options: EvidenceOptions,
    query_options: QueryOptions,
) -> None:
    mode = retrieval_options.mode or settings.retrieval_mode
    if mode == "hybrid" and not plan_context.features.hybrid_allowed:
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=f"当前套餐（{plan_context.plan}）不支持 hybrid 检索",
            context={"plan": plan_context.plan, "profile": profile, "mode": mode},
        )

    if _requested_rerank_enabled(rerank_options) and not plan_context.features.rerank_allowed:
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=f"当前套餐（{plan_context.plan}）不支持 rerank",
            context={"plan": plan_context.plan, "profile": profile},
        )

    if (
        _requested_evidence_enabled(evidence_options)
        and not plan_context.features.evidence_allowed
    ):
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=f"当前套餐（{plan_context.plan}）不支持证据编排",
            context={"plan": plan_context.plan, "profile": profile},
        )

    if (
        _requested_query_rewrite_enabled(query_options)
        and not plan_context.features.query_rewrite_allowed
    ):
        raise AppError(
            ErrorCode.FEATURE_NOT_ALLOWED,
            msg=f"当前套餐（{plan_context.plan}）不支持 query 改写",
            context={"plan": plan_context.plan, "profile": profile},
        )


def _requested_rerank_enabled(rerank_options: RerankOptions) -> bool:
    if rerank_options.enabled is not None:
        return rerank_options.enabled
    return settings.rerank_enabled


def _requested_query_rewrite_enabled(query_options: QueryOptions) -> bool:
    if query_options.enabled is not None:
        if not query_options.enabled:
            return False
    elif not settings.query_rewrite_enabled:
        return False

    return query_options.strategy != "noop"


def _requested_evidence_enabled(evidence_options: EvidenceOptions) -> bool:
    if evidence_options.enabled is not None:
        return evidence_options.enabled
    return settings.evidence_enabled
