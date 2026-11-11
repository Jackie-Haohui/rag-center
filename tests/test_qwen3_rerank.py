import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from app.core.config import settings
from app.providers.rerank.base import RerankProviderError
from app.providers.rerank.qwen3 import Qwen3RerankProvider
from app.schemas.rag import RerankOptions, RetrieveRequest
from app.services.rag_service import RagService


def _chunk(chunk_id: str, *, title: str = "标题", content: str = "候选正文") -> dict:
    return {
        "kb_id": "kb-1",
        "document_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "title": title,
        "content": content,
        "score": 0.5,
        "metadata": {},
    }


def _provider(handler, **overrides) -> Qwen3RerankProvider:
    return Qwen3RerankProvider(
        base_url="https://example.com/compatible-api/v1",
        api_key="test-key",
        model="qwen3-rerank",
        timeout_seconds=1,
        instruct="rank evidence",
        max_candidates=overrides.get("max_candidates", 20),
        chunk_max_chars=overrides.get("chunk_max_chars", 1024),
        transport=httpx.MockTransport(handler),
    )


def test_qwen3_reuses_model_api_key_unless_overridden(monkeypatch) -> None:
    monkeypatch.setattr(settings, "model_api_key", "shared-model-key")
    monkeypatch.setattr(settings, "rerank_api_key", "")
    monkeypatch.setattr(settings, "rerank_model", "configured-rerank-model")

    shared = Qwen3RerankProvider(base_url="https://example.com")
    dedicated = Qwen3RerankProvider(
        base_url="https://example.com",
        api_key="dedicated-rerank-key",
    )

    assert shared.api_key == "shared-model-key"
    assert shared.model == "configured-rerank-model"
    assert dedicated.api_key == "dedicated-rerank-key"


def test_qwen3_maps_indexes_and_sorts_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compatible-api/v1/reranks"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 2, "relevance_score": 0.9},
                ],
                "model": "qwen3-rerank",
            },
        )

    result = asyncio.run(
        _provider(handler).rerank(
            query="问题",
            chunks=[_chunk("a"), _chunk("b"), _chunk("c")],
            top_n=2,
        )
    )

    assert [item["chunk_id"] for item in result] == ["c", "a"]
    assert [item["rerank_score"] for item in result] == [0.9, 0.2]


def test_qwen3_builds_title_content_and_truncates() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    asyncio.run(
        _provider(handler, chunk_max_chars=5).rerank(
            query="问题",
            chunks=[_chunk("a", title="规则", content="123456789")],
            top_n=5,
        )
    )

    assert captured["documents"] == ["标题：规则\n正文：12345"]
    assert captured["top_n"] == 1
    assert captured["instruct"] == "rank evidence"


def test_qwen3_empty_candidates_does_not_call_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    result = asyncio.run(_provider(handler).rerank(query="问题", chunks=[], top_n=5))
    assert result == []


@pytest.mark.parametrize(
    "results, message",
    [
        ([{"index": 2, "relevance_score": 0.5}], "out of range"),
        ([{"relevance_score": 0.5}], "index must be an integer"),
        (
            [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4},
            ],
            "duplicated",
        ),
        ([{"index": 0, "relevance_score": "bad"}], "must be numeric"),
        ([{"index": 0, "relevance_score": "nan"}], "must be finite"),
    ],
)
def test_qwen3_rejects_invalid_results(results, message) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results})

    chunks = [_chunk("a"), _chunk("b")] if len(results) > 1 else [_chunk("a")]
    with pytest.raises(RerankProviderError, match=message):
        asyncio.run(_provider(handler).rerank(query="问题", chunks=chunks, top_n=len(results)))


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_qwen3_rejects_http_errors(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"message": "failed"})

    with pytest.raises(RerankProviderError, match=f"HTTP {status_code}"):
        asyncio.run(_provider(handler).rerank(query="问题", chunks=[_chunk("a")], top_n=1))


def test_qwen3_rejects_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(RerankProviderError, match="timed out"):
        asyncio.run(_provider(handler).rerank(query="问题", chunks=[_chunk("a")], top_n=1))


def test_qwen3_rejects_non_json_and_missing_results() -> None:
    responses = iter(
        [
            httpx.Response(200, content=b"not-json"),
            httpx.Response(200, json={"model": "qwen3-rerank"}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    provider = _provider(handler)
    with pytest.raises(RerankProviderError, match="not valid JSON"):
        asyncio.run(provider.rerank(query="问题", chunks=[_chunk("a")], top_n=1))
    with pytest.raises(RerankProviderError, match="missing results"):
        asyncio.run(provider.rerank(query="问题", chunks=[_chunk("a")], top_n=1))


def test_qwen3_clamps_scores_and_keeps_input_order_on_ties() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 2},
                    {"index": 0, "relevance_score": 2},
                    {"index": 2, "relevance_score": -1},
                ]
            },
        )

    result = asyncio.run(
        _provider(handler).rerank(
            query="问题",
            chunks=[_chunk("a"), _chunk("b"), _chunk("c")],
            top_n=3,
        )
    )

    assert [item["chunk_id"] for item in result] == ["a", "b", "c"]
    assert [item["rerank_score"] for item in result] == [1.0, 1.0, 0.0]


def test_rag_service_records_qwen3_metadata(monkeypatch) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9}]})

    service = RagService(MagicMock())
    provider = _provider(handler)
    monkeypatch.setattr(service, "_build_rerank_provider", lambda: provider)
    request = RetrieveRequest(
        kb_id="kb-1",
        user_id="user",
        query="问题",
        profile="custom",
        rerank_options=RerankOptions(enabled=True, top_n=1),
    )

    chunks, metadata = asyncio.run(
        service._maybe_rerank(
            request=request,
            rerank_query="改写后的问题",
            chunks=[_chunk("a"), _chunk("b")],
            top_k=2,
        )
    )

    assert chunks[0]["chunk_id"] == "b"
    assert captured["query"] == "改写后的问题"
    assert metadata.provider == "qwen3"
    assert metadata.model == "qwen3-rerank"
    assert metadata.candidate_count == 2
    assert metadata.returned_count == 1
    assert metadata.latency_ms is not None
    assert metadata.degraded is False


def test_rag_service_degrades_on_qwen3_failure(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "failed"})

    service = RagService(MagicMock())
    provider = _provider(handler)
    monkeypatch.setattr(service, "_build_rerank_provider", lambda: provider)
    request = RetrieveRequest(
        kb_id="kb-1",
        user_id="user",
        query="问题",
        profile="custom",
        rerank_options=RerankOptions(enabled=True, top_n=1),
    )

    chunks, metadata = asyncio.run(
        service._maybe_rerank(
            request=request,
            rerank_query="改写后的问题",
            chunks=[_chunk("a"), _chunk("b")],
            top_k=2,
        )
    )

    assert [item["chunk_id"] for item in chunks] == ["a"]
    assert chunks[0]["rerank_score"] is None
    assert metadata.provider == "qwen3"
    assert metadata.model == "qwen3-rerank"
    assert metadata.returned_count == 1
    assert metadata.degraded is True
    assert "HTTP 500" in str(metadata.error)
