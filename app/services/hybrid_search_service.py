from app.core.logging import get_logger

logger = get_logger(__name__)


class HybridSearchService:
    def fuse_by_rrf(
        self,
        *,
        vector_chunks: list[dict],
        bm25_chunks: list[dict],
        rrf_k: int,
        top_n: int,
    ) -> list[dict]:
        merged: dict[tuple[str, str], dict] = {}

        for rank, chunk in enumerate(vector_chunks, start=1):
            key = (str(chunk.get("kb_id") or ""), str(chunk["chunk_id"]))
            merged[key] = {
                **chunk,
                "vector_score": chunk.get("score"),
                "bm25_score": None,
                "vector_rank": rank,
                "bm25_rank": None,
            }

        for rank, chunk in enumerate(bm25_chunks, start=1):
            key = (str(chunk.get("kb_id") or ""), str(chunk["chunk_id"]))
            existing = merged.get(key)
            if existing is None:
                merged[key] = {
                    **chunk,
                    "vector_score": None,
                    "bm25_score": chunk.get("bm25_score"),
                    "vector_rank": None,
                    "bm25_rank": rank,
                }
            else:
                existing["bm25_score"] = chunk.get("bm25_score")
                existing["bm25_rank"] = rank
                for field in (
                    "retrieval_text",
                    "index_version",
                    "section_id",
                    "parent_section_id",
                    "order_index",
                    "metadata",
                ):
                    if existing.get(field) is None and chunk.get(field) is not None:
                        existing[field] = chunk[field]

        fused: list[dict] = []
        for chunk in merged.values():
            score = 0.0
            if chunk["vector_rank"] is not None:
                score += 1.0 / (rrf_k + chunk["vector_rank"])
            if chunk["bm25_rank"] is not None:
                score += 1.0 / (rrf_k + chunk["bm25_rank"])

            if chunk["vector_rank"] is not None and chunk["bm25_rank"] is not None:
                retrieval_source = "hybrid"
            elif chunk["vector_rank"] is not None:
                retrieval_source = "vector"
            else:
                retrieval_source = "bm25"

            fused.append(
                {
                    **chunk,
                    "score": score,
                    "retrieval_source": retrieval_source,
                }
            )

        fused.sort(
            key=lambda item: (
                -item["score"],
                -int(item["retrieval_source"] == "hybrid"),
                min(
                    rank
                    for rank in (item.get("vector_rank"), item.get("bm25_rank"))
                    if rank is not None
                ),
                str(item.get("kb_id") or ""),
                str(item.get("chunk_id") or ""),
            )
        )
        results = fused[:top_n]
        logger.info(
            "RRF_FUSION_SUCCESS | vector_count=%s | bm25_count=%s | fused_count=%s | rrf_k=%s | top_n=%s",
            len(vector_chunks),
            len(bm25_chunks),
            len(fused),
            rrf_k,
            top_n,
        )
        return results
