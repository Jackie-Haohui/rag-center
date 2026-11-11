import asyncio
from types import SimpleNamespace

from app.providers.rerank.qwen3 import Qwen3RerankProvider
from app.services.context_expansion_service import ContextExpansionService
from app.services.graph_candidate_expansion_service import GraphCandidateExpansionService
from app.services.retrieval_candidate_filter import filter_retrieval_candidates


def chunk(chunk_id: str, order_index: int, content: str, *, section_id: str = "s1"):
    return SimpleNamespace(
        id=chunk_id,
        tenant_id="tenant",
        kb_id="kb",
        document_id="doc",
        index_version="v2",
        section_id=section_id,
        parent_section_id=None,
        order_index=order_index,
        title="文档",
        content=content,
        retrieval_text=f"文档：文档\n正文：{content}",
        metadata_={"heading_path": "规则/退款", "chunk_type": "prose"},
    )


def anchor() -> dict:
    return {
        "kb_id": "kb",
        "document_id": "doc",
        "chunk_id": "anchor",
        "title": "文档",
        "content": "可以申请退款。",
        "retrieval_text": "文档：文档\n章节：规则 > 退款\n正文：可以申请退款。",
        "index_version": "v2",
        "section_id": "s1",
        "parent_section_id": None,
        "order_index": 2,
        "score": 0.8,
        "rerank_score": 0.9,
        "metadata": {"heading_path": "规则/退款", "chunk_type": "prose"},
    }


def test_candidate_filter_removes_reference_pointer() -> None:
    chunks = [
        {"chunk_id": "a", "content": "有效规则。", "metadata": {}},
        {
            "chunk_id": "b",
            "content": "> 关联规则：参见《退款政策》",
            "metadata": {"chunk_type": "reference_pointer"},
        },
    ]
    result, metadata = filter_retrieval_candidates(chunks)
    assert [item["chunk_id"] for item in result] == ["a"]
    assert metadata["filtered_reasons"]["reference_pointer"] == 1


def test_candidate_filter_removes_low_information_and_duplicates_without_reordering() -> None:
    chunks = [
        {"chunk_id": "a", "content": "有效规则。", "metadata": {}},
        {"chunk_id": "b", "content": "---", "metadata": {}},
        {"chunk_id": "c", "content": "有效规则。", "metadata": {}},
        {"chunk_id": "d", "content": "48 小时内发货。", "metadata": {}},
    ]
    result, metadata = filter_retrieval_candidates(chunks)
    assert [item["chunk_id"] for item in result] == ["a", "d"]
    assert metadata["candidate_count_before_filter"] == 4
    assert metadata["candidate_count_after_filter"] == 2


def test_qwen3_prefers_retrieval_text_and_v1_uses_legacy_text() -> None:
    provider = Qwen3RerankProvider(base_url="https://example.com", api_key="key")
    contextual = provider._build_document(
        {"title": "旧标题", "content": "旧正文", "retrieval_text": "上下文化检索文本"}
    )
    legacy = provider._build_document({"title": "旧标题", "content": "旧正文"})
    assert contextual == "上下文化检索文本"
    assert legacy == "标题：旧标题\n正文：旧正文"


def test_v1_skips_context_expansion() -> None:
    service = ContextExpansionService.__new__(ContextExpansionService)
    result = asyncio.run(
        service.expand(
            tenant_id="tenant",
            kb_ids=["kb"],
            anchors=[{**anchor(), "index_version": "v1"}],
        )
    )
    assert result.anchors[0]["context"] is None
    assert result.metadata.executed is False


def test_v2_expands_neighbors_and_reference_without_changing_anchor_score() -> None:
    service = ContextExpansionService.__new__(ContextExpansionService)
    previous = chunk("previous", 1, "适用条件是尚未发货。")
    following = chunk("next", 3, "审核后两小时原路退回。")
    referenced = chunk("reference", 7, "银行卡一至三个工作日到账。", section_id="s2")

    class ChunkRepository:
        def find_neighbors_batch(self, *args, **kwargs):
            return [following, previous]

        def first_in_sections_batch(self, *args, **kwargs):
            return []

        def get_by_ids_in_allowed_scope(self, ids, **kwargs):
            return [referenced] if ids else []

    class RelationRepository:
        def list_targets_batch(self, *args, **kwargs):
            return {"anchor": ["reference"]}

    service.chunk_repository = ChunkRepository()
    service.relation_repository = RelationRepository()
    original = anchor()
    result = asyncio.run(
        service.expand(tenant_id="tenant", kb_ids=["kb"], anchors=[original])
    )
    expanded = result.anchors[0]

    assert expanded["score"] == original["score"]
    assert expanded["rerank_score"] == original["rerank_score"]
    assert [source["relation"] for source in expanded["context"]["sources"]] == [
        "anchor",
        "reference",
        "previous",
        "next",
    ]
    assert result.metadata.supplemental_chunk_count == 3


def test_graph_injection_appends_reference_without_duplicating_seeds() -> None:
    service = GraphCandidateExpansionService.__new__(GraphCandidateExpansionService)
    referenced = chunk("reference", 7, "银行卡一至三个工作日到账。", section_id="s2")

    class ChunkRepository:
        def find_neighbors_batch(self, *args, **kwargs):
            return []

        def first_in_sections_batch(self, *args, **kwargs):
            return []

        def get_by_ids_in_allowed_scope(self, ids, **kwargs):
            return [referenced] if ids else []

    class RelationRepository:
        def list_targets_batch(self, *args, **kwargs):
            return {"anchor": ["reference"]}

    service.chunk_repository = ChunkRepository()
    service.relation_repository = RelationRepository()
    seed = anchor()
    result = asyncio.run(
        service.expand_candidates(
            tenant_id="tenant",
            kb_ids=["kb"],
            seeds=[seed],
        )
    )
    assert result.metadata.executed is True
    assert result.metadata.injected_count == 1
    assert [item["chunk_id"] for item in result.candidates] == ["anchor", "reference"]
    assert result.candidates[1]["metadata"]["injection_source"] == "reference"


def test_context_query_failure_degrades_to_original_anchor() -> None:
    service = ContextExpansionService.__new__(ContextExpansionService)

    class BrokenRepository:
        def find_neighbors_batch(self, *args, **kwargs):
            raise RuntimeError("relation store unavailable")

    service.chunk_repository = BrokenRepository()
    service.relation_repository = SimpleNamespace()
    original = anchor()
    result = asyncio.run(
        service.expand(tenant_id="tenant", kb_ids=["kb"], anchors=[original])
    )
    assert result.anchors[0]["chunk_id"] == "anchor"
    assert result.anchors[0]["context"] is None
    assert result.metadata.degraded is True
