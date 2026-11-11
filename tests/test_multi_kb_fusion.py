from app.services.hybrid_search_service import HybridSearchService


def test_global_rrf_does_not_reserve_equal_library_slots():
    vector = [
        {"kb_id": "kb-a", "chunk_id": "a1", "score": 0.99},
        {"kb_id": "kb-a", "chunk_id": "a2", "score": 0.98},
        {"kb_id": "kb-b", "chunk_id": "b1", "score": 0.50},
    ]
    bm25 = [
        {"kb_id": "kb-a", "chunk_id": "a1", "bm25_score": 10.0},
        {"kb_id": "kb-a", "chunk_id": "a2", "bm25_score": 9.0},
        {"kb_id": "kb-b", "chunk_id": "b1", "bm25_score": 1.0},
    ]

    fused = HybridSearchService().fuse_by_rrf(
        vector_chunks=vector,
        bm25_chunks=bm25,
        rrf_k=60,
        top_n=2,
    )

    assert [(item["kb_id"], item["chunk_id"]) for item in fused] == [
        ("kb-a", "a1"),
        ("kb-a", "a2"),
    ]


def test_global_rrf_uses_kb_and_chunk_as_unique_key():
    vector = [
        {"kb_id": "kb-b", "chunk_id": "same", "score": 0.9},
        {"kb_id": "kb-a", "chunk_id": "same", "score": 0.9},
    ]

    fused = HybridSearchService().fuse_by_rrf(
        vector_chunks=vector,
        bm25_chunks=[],
        rrf_k=60,
        top_n=2,
    )

    assert len(fused) == 2
    assert {(item["kb_id"], item["chunk_id"]) for item in fused} == {
        ("kb-a", "same"),
        ("kb-b", "same"),
    }
