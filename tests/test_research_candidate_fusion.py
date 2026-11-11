from app.schemas.rag import (
    RetrieveContextMetadata,
    RetrieveGraphInjectionMetadata,
    RetrieveRerankMetadata,
)
from app.schemas.research import ResearchAspect
from app.services.research_candidate_fusion_service import (
    ResearchCandidateFusionService,
)
from app.services.retrieve_once_service import RetrieveOnceResult


def _chunk(
    chunk_id,
    content,
    *,
    kb_id="kb-1",
    source="hybrid",
    context=None,
):
    return {
        "chunk_id": chunk_id,
        "kb_id": kb_id,
        "document_id": "doc-1",
        "index_version": "v2",
        "title": "规则",
        "content": content,
        "retrieval_source": source,
        "context": context,
        "metadata": {},
    }


def _retrieval(query_id, aspects, chunks, round=1):
    return RetrieveOnceResult(
        query_id=query_id,
        query=query_id,
        aspect_ids=aspects,
        round=round,
        candidate_snapshot=chunks,
        retrieved_chunks=chunks,
        retrieval_metadata={},
        candidate_filter_metadata={},
        graph_injection_metadata=RetrieveGraphInjectionMetadata(),
        context_metadata=RetrieveContextMetadata(),
        rerank_metadata=RetrieveRerankMetadata(enabled=False, provider="none"),
        latency_ms=1,
    )


def test_fusion_deduplicates_chunk_and_merges_provenance_with_rrf():
    shared = _chunk("shared", "同一规则")
    result = ResearchCandidateFusionService().fuse(
        results=[
            _retrieval("Q1", ["A1"], [shared]),
            _retrieval("Q2", ["A2"], [shared], round=2),
        ],
        aspects=[
            ResearchAspect(aspect_id="A1", description="积分"),
            ResearchAspect(aspect_id="A2", description="现金"),
        ],
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.candidate_id == "C1"
    assert candidate.query_ids == ["Q1", "Q2"]
    assert candidate.aspect_ids == ["A1", "A2"]
    assert candidate.rounds == [1, 2]
    assert candidate.fusion_score == 2 / 61


def test_same_content_in_different_knowledge_bases_is_preserved():
    result = ResearchCandidateFusionService().fuse(
        results=[
            _retrieval("Q1", ["A1"], [_chunk("c1", "同文", kb_id="kb-1")]),
            _retrieval("Q2", ["A1"], [_chunk("c2", "同文", kb_id="kb-2")]),
        ],
        aspects=[ResearchAspect(aspect_id="A1", description="要点")],
    )
    assert len(result.candidates) == 2
    assert {item.kb_id for item in result.candidates} == {"kb-1", "kb-2"}


def test_fair_selection_reserves_candidates_for_each_aspect():
    a1_chunks = [_chunk(f"a1-{index}", f"高频候选{index}") for index in range(5)]
    a2_chunks = [_chunk("a2-1", "低频但必要")]
    result = ResearchCandidateFusionService().fuse(
        results=[
            _retrieval("Q1", ["A1"], a1_chunks),
            _retrieval("Q2", ["A2"], a2_chunks),
        ],
        aspects=[
            ResearchAspect(aspect_id="A1", description="高频"),
            ResearchAspect(aspect_id="A2", description="必要"),
        ],
        max_candidates=2,
    )
    assert {item.chunk_id for item in result.candidates} == {"a1-0", "a2-1"}
    assert result.aspect_candidate_ids["A2"]


def test_context_source_and_graph_source_are_traceable_and_bounded():
    context = {
        "sources": [
            {"chunk_id": "anchor", "relation": "anchor", "content": "锚点"},
            {
                "chunk_id": "reference",
                "relation": "reference",
                "content": "引用规则",
                "document_id": "doc-2",
                "title": "引用",
            },
        ]
    }
    chunks = [
        _chunk("anchor", "锚点", context=context),
        _chunk("graph", "图谱候选", source="graph_injection"),
    ]
    result = ResearchCandidateFusionService().fuse(
        results=[_retrieval("Q1", ["A1"], chunks)],
        aspects=[ResearchAspect(aspect_id="A1", description="要点")],
        max_candidates=3,
    )

    assert len(result.candidates) == 3
    assert any(item.source == "context:reference" for item in result.candidates)
    assert any("graph_injection" in item.metadata["sources"] for item in result.candidates)
    context_candidate = next(item for item in result.candidates if item.source.startswith("context"))
    assert context_candidate.metadata["anchor_chunk_ids"] == ["anchor"]


def test_empty_aspect_is_retained_in_mapping():
    result = ResearchCandidateFusionService().fuse(
        results=[_retrieval("Q1", ["A1"], [_chunk("c1", "只有要点一")])],
        aspects=[
            ResearchAspect(aspect_id="A1", description="要点一"),
            ResearchAspect(aspect_id="A2", description="要点二"),
        ],
    )
    assert result.aspect_candidate_ids["A2"] == []
