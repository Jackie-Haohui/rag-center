import asyncio

from app.providers.query.base import KnowledgeBaseQueryContext
from app.providers.query.pipeline import QueryProcessorPipeline
from app.schemas.rag import QueryOptions


def _kb() -> KnowledgeBaseQueryContext:
    return KnowledgeBaseQueryContext(
        name="电商知识库",
        description=None,
        settings={
            "synonyms": [
                {
                    "terms": ["价保"],
                    "expand": ["价格保护", "退差价"],
                }
            ]
        },
    )


def test_synonym_expansion_can_be_disabled_per_request():
    result = asyncio.run(
        QueryProcessorPipeline().process(
            "这个商品支持价保吗",
            kb=_kb(),
            query_options=QueryOptions(
                enabled=False,
                strategy="noop",
                synonym_enabled=False,
            ),
        )
    )

    assert result.search_query == "这个商品支持价保吗"
    assert result.synonym_enabled is False
    assert result.synonym_applied is False
    assert result.synonym_control_explicit is True


def test_synonym_expansion_can_be_enabled_per_request():
    result = asyncio.run(
        QueryProcessorPipeline().process(
            "这个商品支持价保吗",
            kb=_kb(),
            query_options=QueryOptions(
                enabled=False,
                strategy="noop",
                synonym_enabled=True,
            ),
        )
    )

    assert result.search_query == "这个商品支持价保吗 价格保护 退差价"
    assert result.synonym_enabled is True
    assert result.synonym_applied is True
    assert result.synonym_control_explicit is True


def test_synonym_expansion_defaults_to_disabled_when_unspecified():
    result = asyncio.run(
        QueryProcessorPipeline().process(
            "这个商品支持价保吗",
            kb=_kb(),
            query_options=QueryOptions(enabled=False, strategy="noop"),
        )
    )

    assert result.search_query == "这个商品支持价保吗"
    assert result.synonym_enabled is False
    assert result.synonym_applied is False
    assert result.synonym_control_explicit is False


def test_synonym_expansion_defaults_to_disabled_without_query_options():
    result = asyncio.run(
        QueryProcessorPipeline().process(
            "这个商品支持价保吗",
            kb=_kb(),
            query_options=None,
        )
    )

    assert result.search_query == "这个商品支持价保吗"
    assert result.synonym_enabled is False
    assert result.synonym_applied is False
    assert result.synonym_control_explicit is False
