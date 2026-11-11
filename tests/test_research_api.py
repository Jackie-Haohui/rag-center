import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.v1.routes import rag as rag_routes
from app.schemas.research import ResearchRequest


def test_research_request_reuses_kb_normalization_and_has_no_retrieval_controls():
    request = ResearchRequest(
        kb_ids=[" kb-1 ", "kb-1", "kb-2"],
        user_id="user",
        query=" 退款规则 ",
    )
    assert request.kb_id == "kb-1"
    assert request.kb_ids == ["kb-1", "kb-2"]
    assert request.query == "退款规则"
    assert not hasattr(request, "profile")
    assert not hasattr(request, "top_k")


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "speed"),
        ("index_version", "v1"),
        ("retrieval_options", {"mode": "vector"}),
        ("rerank_options", {"enabled": False}),
        ("top_k", 10),
        ("max_rounds", 9),
        ("tools", ["browser"]),
        ("web_search", True),
    ],
)
def test_research_request_rejects_limits_and_external_tool_controls(field, value):
    payload = {"kb_id": "kb", "user_id": "user", "query": "问题", field: value}
    with pytest.raises(ValidationError):
        ResearchRequest(**payload)


def test_research_route_uses_dedicated_service(monkeypatch):
    expected = SimpleNamespace(model_dump=lambda **kwargs: {"query": "问题"})
    clients = SimpleNamespace()

    class FakeService:
        def __init__(self, db, *, http_clients):
            self.db = db
            assert http_clients is clients

        async def research(self, tenant_id, request):
            assert tenant_id == "tenant"
            assert request.query == "问题"
            return expected

    monkeypatch.setattr(rag_routes, "ResearchService", FakeService)
    response = asyncio.run(
        rag_routes.research(
            ResearchRequest(kb_id="kb", user_id="user", query="问题"),
            db=SimpleNamespace(),
            tenant=SimpleNamespace(tenant_id="tenant"),
            http_clients=clients,
        )
    )
    assert response["code"] == 0
    assert response["data"] == {"query": "问题"}
