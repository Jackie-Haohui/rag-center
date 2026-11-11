import asyncio
import json

import httpx

from app.core.config import settings
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.schemas.research import ResearchState, normalize_research_query
from app.services import research_limits


def test_llm_role_router_uses_fast_and_strong_models(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "fast-model")
    monkeypatch.setattr(settings, "llm_strong_model", "strong-model")

    assert LLMRoleRouter.model_for(LLMRole.FAST) == "fast-model"
    assert LLMRoleRouter.model_for(LLMRole.STRONG) == "strong-model"


def test_chat_json_supports_thinking_and_max_tokens():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="fast-model",
    )

    original_async_client = httpx.AsyncClient

    class TransportClient(original_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = TransportClient
    try:
        result = asyncio.run(
            provider.chat_json(
                system_prompt="system",
                user_payload={"value": 1},
                max_tokens=123,
                thinking="disabled",
                log_payload=False,
            )
        )
    finally:
        httpx.AsyncClient = original_async_client

    assert result == {"ok": True}
    assert captured["max_tokens"] == 123
    assert captured["thinking"] == {"type": "disabled"}
    assert provider.last_usage["total_tokens"] == 5


def test_research_state_normalizes_queries_and_accumulates_usage():
    state = ResearchState(
        research_id="research-1",
        original_query="Question",
        tenant_id="tenant",
        kb_ids=["kb"],
        index_versions={"kb": "v2"},
    )
    state.mark_visited("  Hello   WORLD？ ")
    state.mark_visited("hello world")
    state.record_usage({"prompt_tokens": 10, "completion_tokens": 4})

    assert state.visited_queries == ["hello world"]
    assert state.has_visited("HELLO WORLD!") is True
    assert state.input_tokens == 10
    assert state.output_tokens == 4
    assert normalize_research_query(" A  B。 ") == "a b"


def test_research_limits_are_bounded():
    assert research_limits.MAX_INITIAL_QUERIES == 3
    assert research_limits.MAX_SUPPLEMENT_QUERIES == 2
    assert research_limits.MAX_RESEARCH_ROUNDS == 2
    assert research_limits.MAX_RESEARCH_LLM_CALLS == 3
    assert research_limits.DEFAULT_RESEARCH_TIMEOUT_SECONDS == 90
