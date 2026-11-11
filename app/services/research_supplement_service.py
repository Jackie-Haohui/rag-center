from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.index_versions import DEFAULT_INDEX_VERSION
from app.schemas.rag import RetrieveRequest
from app.schemas.research import PlannedQuery, ResearchCandidate, ResearchState
from app.services.research_limits import MAX_SUPPLEMENT_QUERIES
from app.services.research_parallel_retrieval_service import (
    ParallelRetrievalResult,
    ResearchParallelRetrievalService,
)


@dataclass(slots=True)
class SupplementResult:
    retrieval: ParallelRetrievalResult
    new_chunk_count: int


class ResearchSupplementService:
    def __init__(self, parallel_retrieval: ResearchParallelRetrievalService) -> None:
        self.parallel_retrieval = parallel_retrieval

    async def execute(
        self,
        *,
        state: ResearchState,
        queries: list[PlannedQuery],
        request: RetrieveRequest,
        knowledge_bases: list[Any],
        existing_candidates: list[ResearchCandidate],
    ) -> SupplementResult:
        retrieval = await self.parallel_retrieval.execute(
            state=state,
            queries=queries[:MAX_SUPPLEMENT_QUERIES],
            request=request,
            knowledge_bases=knowledge_bases,
            round=2,
        )
        existing_keys = {
            (candidate.kb_id, candidate.index_version, candidate.chunk_id)
            for candidate in existing_candidates
        }
        new_keys = {
            (
                str(chunk.get("kb_id") or ""),
                str(chunk.get("index_version") or DEFAULT_INDEX_VERSION),
                str(chunk.get("chunk_id") or ""),
            )
            for result in retrieval.results
            for chunk in result.retrieved_chunks
            if chunk.get("chunk_id")
        }
        new_chunk_count = len(new_keys - existing_keys)
        return SupplementResult(retrieval=retrieval, new_chunk_count=new_chunk_count)
