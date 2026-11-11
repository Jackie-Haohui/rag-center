import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.indexing_service import IndexingService
from app.services.reference_relation_service import ParsedReference, ReferenceRelationService
from app.utils.markdown_splitter import (
    MarkdownStructuredSplitter,
    build_retrieval_text,
    is_low_information_content,
)


def make_splitter(**overrides) -> MarkdownStructuredSplitter:
    values = {"chunk_size": 200, "chunk_overlap": 20, "table_max_rows_per_chunk": 2}
    values.update(overrides)
    return MarkdownStructuredSplitter(**values)


def test_low_information_filter_keeps_meaningful_short_rules() -> None:
    for value in ("", "---", "***", "___", "!!!", "1."):
        assert is_low_information_content(value)
    assert is_low_information_content("## 退款规则", "退款规则")
    assert not is_low_information_content("一个订单只能使用一张优惠券")
    assert not is_low_information_content("48 小时内发货")


def test_contextual_split_assigns_hierarchy_and_continuous_order() -> None:
    pieces = make_splitter().split(
        """# 平台规则

总则正文。

## 售后

售后说明。

### 退款

退款申请通过后原路退回。
""",
        document_id="doc-1",
        index_version="v2",
    )

    assert [piece.metadata["order_index"] for piece in pieces] == list(range(len(pieces)))
    refund = next(piece for piece in pieces if piece.metadata["heading_path"].endswith("退款"))
    aftersales = next(piece for piece in pieces if piece.metadata["heading_path"].endswith("售后"))
    assert refund.metadata["parent_section_id"] == aftersales.metadata["section_id"]
    assert refund.text == "退款申请通过后原路退回。"
    assert "### 退款" not in refund.text


def test_duplicate_body_in_same_section_is_removed() -> None:
    pieces = make_splitter().split(
        "## 规则\n\n同一规则。\n\n同一规则。",
        document_id="doc-1",
        index_version="v2",
    )
    assert len(pieces) == 1


def test_table_and_list_keep_section_context() -> None:
    pieces = make_splitter().split(
        """## 退款

- 银行卡原路退回
- 积分退回积分池

| 渠道 | 时效 |
|---|---|
| 银行卡 | 1~3 天 |
| 余额 | 即时 |
| 微信 | 1~3 天 |
""",
        document_id="doc-1",
        index_version="v2",
    )
    assert len(pieces) == 1
    assert pieces[0].metadata["heading_path"] == "退款"
    assert "银行卡原路退回" in pieces[0].text
    assert "【表头】渠道 | 时效" in pieces[0].text


def test_oversized_section_keeps_table_header_when_split() -> None:
    pieces = make_splitter(chunk_size=50, table_max_rows_per_chunk=1).split(
        """## 退款

| 渠道 | 时效 |
|---|---|
| 银行卡 | 1~3 天 |
| 余额 | 即时 |
| 微信 | 1~3 天 |
""",
        document_id="doc-1",
        index_version="v2",
    )
    table_pieces = [piece for piece in pieces if "【表头】渠道 | 时效" in piece.text]
    assert len(table_pieces) >= 2
    assert all(piece.metadata["heading_path"] == "退款" for piece in table_pieces)


def test_retrieval_text_separates_title_path_and_content() -> None:
    value = build_retrieval_text(
        title="售后政策.md",
        heading_path="售后/退款",
        content="银行卡 1~3 个工作日到账。",
    )
    assert value == "文档：售后政策.md\n章节：售后 > 退款\n正文：银行卡 1~3 个工作日到账。"


def test_reference_pointer_merges_into_same_section_evidence() -> None:
    pieces = make_splitter().split(
        """## 退款

退款申请通过后原路退回。

> 关联规则：现金部分的到账路径参见《支付方式与资金安全规范》第 3 节“退款原路退回”。
""",
        document_id="doc-1",
        index_version="v2",
    )
    assert len(pieces) == 1
    assert pieces[0].metadata.get("chunk_type") != "reference_pointer"
    assert "退款申请通过后原路退回。" in pieces[0].text
    assert pieces[0].text.count("关联规则") == 1
    assert any("支付方式与资金安全规范" in text for text in pieces[0].metadata.get("reference_texts", []))


def test_section_label_merges_with_following_list() -> None:
    pieces = make_splitter().split(
        """## 投放准入

**准入：**

- 店铺状态正常
- 推广商品须已上架
""",
        document_id="doc-1",
        index_version="v2",
    )
    assert len(pieces) == 1
    assert "**准入：**" in pieces[0].text
    assert "店铺状态正常" in pieces[0].text


def test_indexing_v2_keeps_reference_in_searchable_section_chunk() -> None:
    class Embeddings:
        def __init__(self) -> None:
            self.inputs = []

        async def embed_texts(self, texts):
            self.inputs = texts
            return [[0.1, 0.2] for _ in texts]

    class Store:
        def __init__(self) -> None:
            self.chunks = []

        async def add_chunks(self, chunks):
            self.chunks = chunks

    class ChunkRepo:
        def __init__(self) -> None:
            self.chunks = []

        def add_many(self, chunks):
            self.chunks = chunks

    embeddings = Embeddings()
    store = Store()
    chunk_repo = ChunkRepo()
    db = MagicMock()
    service = IndexingService(
        db,
        embedding_provider=embeddings,
        vector_store=store,
        text_splitter=make_splitter(),
    )
    service.chunk_repository = chunk_repo
    document = SimpleNamespace(
        id="doc-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        title="退款政策.md",
        content=(
            "## 退款\n\n退款申请通过后原路退回。\n\n"
            "> 关联规则：现金部分的到账路径参见《支付方式与资金安全规范》第 3 节“退款原路退回”。"
        ),
        source_type="markdown",
    )

    asyncio.run(service.index_document_version(document, index_version="v2"))

    assert len(embeddings.inputs) == 1
    assert len(store.chunks) == 1
    assert "退款申请通过后原路退回。" in store.chunks[0]["content"]
    assert "关联规则" in store.chunks[0]["content"]
    assert chunk_repo.chunks == []
    assert store.chunks[0]["metadata"]["chunk_type"] != "reference_pointer"


def test_indexing_v2_embeds_retrieval_text_and_keeps_raw_content() -> None:
    class Embeddings:
        def __init__(self) -> None:
            self.inputs = []

        async def embed_texts(self, texts):
            self.inputs = texts
            return [[0.1, 0.2] for _ in texts]

    class Store:
        def __init__(self) -> None:
            self.chunks = []

        async def add_chunks(self, chunks):
            self.chunks = chunks

    embeddings = Embeddings()
    store = Store()
    service = IndexingService(
        MagicMock(),
        embedding_provider=embeddings,
        vector_store=store,
        text_splitter=make_splitter(),
    )
    document = SimpleNamespace(
        id="doc-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        title="退款政策.md",
        content="## 到账\n\n银行卡 1~3 个工作日到账。",
        source_type="markdown",
    )

    asyncio.run(service.index_document_version(document, index_version="v2"))

    assert embeddings.inputs[0].startswith("文档：退款政策.md\n章节：到账\n正文：")
    assert store.chunks[0]["content"] == "银行卡 1~3 个工作日到账。"
    assert store.chunks[0]["retrieval_text"] == embeddings.inputs[0]


def test_reference_target_requires_unique_section() -> None:
    reference = ParsedReference("原文", "文档", "3", "退款")
    one = SimpleNamespace(
        id="a",
        section_id="section-a",
        order_index=1,
        metadata_={"heading_path": "规则/退款"},
    )
    duplicate = SimpleNamespace(
        id="b",
        section_id="section-b",
        order_index=2,
        metadata_={"heading_path": "其他/退款"},
    )
    assert ReferenceRelationService._resolve_target_chunk([one], reference) is one
    assert ReferenceRelationService._resolve_target_chunk([one, duplicate], reference) is None
