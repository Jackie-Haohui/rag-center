import pytest
from pydantic import ValidationError

from app.schemas.rag import RetrieveRequest


def test_kb_id_only_normalizes_to_kb_ids():
    request = RetrieveRequest(
        kb_id="kb-001",
        user_id="user",
        query="test",
        profile="balanced",
    )
    assert request.kb_ids == ["kb-001"]
    assert request.kb_id == "kb-001"


def test_kb_ids_takes_priority_over_kb_id():
    request = RetrieveRequest(
        kb_id="ignored",
        kb_ids=["kb-a", "kb-b"],
        user_id="user",
        query="test",
        profile="balanced",
    )
    assert request.kb_ids == ["kb-a", "kb-b"]
    assert request.kb_id == "kb-a"


def test_kb_ids_deduplicates():
    request = RetrieveRequest(
        kb_ids=["kb-a", "kb-a", "kb-b"],
        user_id="user",
        query="test",
        profile="balanced",
    )
    assert request.kb_ids == ["kb-a", "kb-b"]


def test_missing_kb_scope_raises():
    with pytest.raises(ValidationError):
        RetrieveRequest(user_id="user", query="test", profile="balanced")


def test_kb_ids_max_limit():
    with pytest.raises(ValidationError):
        RetrieveRequest(
            kb_ids=[f"kb-{index}" for index in range(6)],
            user_id="user",
            query="test",
            profile="balanced",
        )


def test_blank_query_raises():
    with pytest.raises(ValidationError, match="query 不能为空"):
        RetrieveRequest(
            kb_id="kb-001",
            user_id="user",
            query="   ",
            profile="balanced",
        )


def test_query_max_length_raises():
    with pytest.raises(ValidationError, match="query 长度不能超过"):
        RetrieveRequest(
            kb_id="kb-001",
            user_id="user",
            query="x" * 2001,
            profile="balanced",
        )


def test_query_is_stripped():
    request = RetrieveRequest(
        kb_id="kb-001",
        user_id="user",
        query="  hello  ",
        profile="balanced",
    )
    assert request.query == "hello"


def test_index_version_is_optional_and_validated():
    request = RetrieveRequest(
        kb_id="kb-001",
        user_id="user",
        query="test",
        index_version="v2",
        profile="balanced",
    )
    assert request.index_version == "v2"

    with pytest.raises(ValidationError):
        RetrieveRequest(
            kb_id="kb-001",
            user_id="user",
            query="test",
            index_version="latest",
            profile="balanced",
        )


def test_evaluation_request_can_disable_observability_and_synonyms():
    request = RetrieveRequest(
        kb_id="kb-001",
        user_id="eval_runner",
        query="价保怎么申请",
        profile="custom",
        observability_enabled=False,
        query_options={
            "enabled": False,
            "strategy": "noop",
            "synonym_enabled": False,
        },
    )

    assert request.observability_enabled is False
    assert request.query_options is not None
    assert request.query_options.synonym_enabled is False
