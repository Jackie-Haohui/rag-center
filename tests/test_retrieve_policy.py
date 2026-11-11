import pytest

from app.core.exceptions import AppError, ErrorCode
from app.schemas.rag import RetrieveRequest
from app.tenant.plan_presets import PLAN_PRESETS
from app.tenant.plan_resolver import PlanContext
from app.tenant.retrieve_policy import resolve_retrieve_request


def test_free_plan_rejects_multi_kb_retrieve():
    plan_context = PlanContext(
        plan="free",
        features=PLAN_PRESETS["free"].features,
        limits=PLAN_PRESETS["free"].limits,
    )
    request = RetrieveRequest(
        kb_ids=["kb-a", "kb-b"],
        user_id="user",
        query="test",
        profile="speed",
    )
    with pytest.raises(AppError) as exc_info:
        resolve_retrieve_request(request, plan_context)
    assert exc_info.value.code == ErrorCode.FEATURE_NOT_ALLOWED.code


def test_standard_plan_allows_three_kbs():
    plan_context = PlanContext(
        plan="standard",
        features=PLAN_PRESETS["standard"].features,
        limits=PLAN_PRESETS["standard"].limits,
    )
    request = RetrieveRequest(
        kb_ids=["kb-a", "kb-b", "kb-c"],
        user_id="user",
        query="test",
        profile="balanced",
    )
    resolved, policy = resolve_retrieve_request(request, plan_context)
    assert resolved.kb_ids == ["kb-a", "kb-b", "kb-c"]
    assert policy.max_kb_per_retrieve == 3
    assert policy.actual_kb_count == 3


@pytest.mark.parametrize(
    (
        "profile",
        "top_k",
        "mode",
        "vector_top_k",
        "bm25_top_k",
        "rerank",
        "top_n",
        "rewrite",
    ),
    [
        ("speed", 5, "vector", 10, None, False, None, False),
        ("balanced", 5, "hybrid", 10, 10, False, None, False),
        ("quality", 5, "hybrid", 20, 20, True, 5, True),
    ],
)
def test_named_profiles_resolve_to_fixed_contract(
    profile,
    top_k,
    mode,
    vector_top_k,
    bm25_top_k,
    rerank,
    top_n,
    rewrite,
):
    plan_context = PlanContext(
        plan="pro",
        features=PLAN_PRESETS["pro"].features,
        limits=PLAN_PRESETS["pro"].limits,
    )
    request = RetrieveRequest(
        kb_id="kb-a",
        user_id="user",
        query="test",
        profile=profile,
    )

    resolved, _ = resolve_retrieve_request(request, plan_context)

    assert resolved.top_k == top_k
    assert resolved.retrieval_options.mode == mode
    assert resolved.retrieval_options.vector_top_k == vector_top_k
    assert resolved.retrieval_options.bm25_top_k == bm25_top_k
    assert resolved.retrieval_options.rrf_k == 60
    assert resolved.rerank_options.enabled is rerank
    assert resolved.rerank_options.top_n == top_n
    assert resolved.query_options.enabled is rewrite
    assert resolved.query_options.synonym_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 10),
        ("retrieval_options", {"mode": "hybrid"}),
        ("rerank_options", {"enabled": False}),
        ("evidence_options", {"enabled": False}),
        ("query_options", {"enabled": False}),
    ],
)
def test_named_profile_rejects_any_explicit_advanced_field(field, value):
    plan_context = PlanContext(
        plan="pro",
        features=PLAN_PRESETS["pro"].features,
        limits=PLAN_PRESETS["pro"].limits,
    )
    request = RetrieveRequest(
        kb_id="kb-a",
        user_id="user",
        query="test",
        profile="balanced",
        **{field: value},
    )

    with pytest.raises(AppError) as exc_info:
        resolve_retrieve_request(request, plan_context)

    assert exc_info.value.code == ErrorCode.PARAM_ERROR.code
    assert exc_info.value.data == {
        "profile": "balanced",
        "conflicting_fields": [field],
    }


def test_custom_rrf_k_is_fixed_by_server():
    plan_context = PlanContext(
        plan="pro",
        features=PLAN_PRESETS["pro"].features,
        limits=PLAN_PRESETS["pro"].limits,
    )
    request = RetrieveRequest(
        kb_id="kb-a",
        user_id="user",
        query="test",
        profile="custom",
        retrieval_options={"mode": "hybrid", "rrf_k": 3},
    )

    resolved, _ = resolve_retrieve_request(request, plan_context)

    assert resolved.retrieval_options.rrf_k == 60


def test_named_profile_reports_system_disabled_capability(monkeypatch):
    monkeypatch.setattr("app.tenant.retrieve_policy.settings.rerank_enabled", False)
    monkeypatch.setattr("app.tenant.retrieve_policy.settings.query_rewrite_enabled", False)
    plan_context = PlanContext(
        plan="pro",
        features=PLAN_PRESETS["pro"].features,
        limits=PLAN_PRESETS["pro"].limits,
    )
    request = RetrieveRequest(
        kb_id="kb-a",
        user_id="user",
        query="test",
        profile="quality",
    )

    resolved, policy = resolve_retrieve_request(request, plan_context)

    assert resolved.rerank_options.enabled is False
    assert resolved.query_options.enabled is False
    assert policy.rerank_disabled_reason == "disabled_by_system"
    assert policy.query_rewrite_disabled_reason == "disabled_by_system"
