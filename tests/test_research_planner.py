import asyncio
from types import SimpleNamespace

import pytest

from app.services.research_planner_service import ResearchPlannerService


class FakePlannerProvider:
    model = "deepseek-v4-flash"

    def __init__(self, output=None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.last_usage = {"prompt_tokens": 12, "completion_tokens": 6}
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.output


def _plan(output):
    provider = FakePlannerProvider(output)
    result = asyncio.run(
        ResearchPlannerService(provider).plan(
            question="积分和现金分别怎么退？",
            knowledge_bases=[SimpleNamespace(name="交易规则", description="退款规则")],
        )
    )
    return provider, result


def test_planner_builds_bounded_ids_and_uses_fast_non_thinking_call():
    provider, result = _plan(
        {
            "aspects": ["积分退回", "现金退回"],
            "queries": [
                {"query": "积分退款规则", "aspect_indexes": [0]},
                {"query": "现金退款规则", "aspect_indexes": [1]},
            ],
            "kb_ids": ["untrusted-kb"],
        }
    )

    assert [item.aspect_id for item in result.aspects] == ["A1", "A2"]
    assert [item.query_id for item in result.queries] == ["Q1", "Q2"]
    assert result.queries[1].aspect_ids == ["A2"]
    assert result.degraded is False
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 6}
    assert provider.calls[0]["thinking"] == "disabled"
    assert provider.calls[0]["max_tokens"] == 800
    assert provider.calls[0]["log_payload"] is False
    assert provider.calls[0]["user_payload"]["knowledge_bases"] == [
        {"name": "交易规则", "description": "退款规则"}
    ]


def test_one_query_can_cover_multiple_aspects():
    _, result = _plan(
        {
            "aspects": ["积分退回", "现金退回"],
            "queries": [{"query": "积分与现金退款规则", "aspect_indexes": [0, 1]}],
        }
    )
    assert result.queries[0].aspect_ids == ["A1", "A2"]


def test_dependent_plan_only_retrieves_premises_in_first_round():
    _, result = _plan(
        {
            "problem_type": "dependent",
            "aspects": ["确认岗位职级与差旅等级", "确认城市类别", "确认住宿上限"],
            "prerequisite_aspect_indexes": [0, 1],
            "queries": [
                {"query": "研发经理对应的职级和差旅等级", "aspect_indexes": [0]},
                {"query": "苏州工业园区差旅城市类别", "aspect_indexes": [1]},
            ],
        }
    )

    assert result.problem_type == "dependent"
    assert [item.stage for item in result.aspects] == ["premise", "premise", "answer"]
    assert [item.aspect_ids for item in result.queries] == [["A1"], ["A2"]]


@pytest.mark.parametrize(
    "output",
    [
        {"aspects": ["积分", "积分"], "queries": [{"query": "退款", "aspect_indexes": [0]}]},
        {"aspects": ["积分"], "queries": [{"query": "退款", "aspect_indexes": [1]}]},
        {
            "aspects": ["积分", "现金"],
            "queries": [{"query": "只查积分", "aspect_indexes": [0]}],
        },
        {"aspects": ["积分"], "queries": [{"query": "https://unsafe", "aspect_indexes": [0]}]},
        {"aspects": ["积分"], "queries": [{"query": "x" * 301, "aspect_indexes": [0]}]},
    ],
)
def test_invalid_planner_output_falls_back_to_original_question(output):
    provider = FakePlannerProvider(output)
    result = asyncio.run(
        ResearchPlannerService(provider).plan(question="原问题", knowledge_bases=[])
    )

    assert result.degraded is True
    assert result.queries[0].query == "原问题"
    assert result.queries[0].aspect_ids == ["A1"]
    assert len(result.queries) == 1


def test_planner_provider_failure_falls_back_without_leaking_error_text():
    provider = FakePlannerProvider(error=RuntimeError("secret-key-value"))
    result = asyncio.run(
        ResearchPlannerService(provider).plan(question="原问题", knowledge_bases=[])
    )

    assert result.degraded is True
    assert result.error == "RuntimeError"
    assert "secret" not in result.error
