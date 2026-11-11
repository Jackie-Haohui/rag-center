from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.core.logging import get_logger
from app.schemas.rag import RetrieveRequest
from app.schemas.research import PlannedQuery, ResearchState, ResearchTaskResult
from app.services.research_limits import MAX_INITIAL_QUERIES, MAX_SUPPLEMENT_QUERIES
from app.services.retrieve_once_service import RetrieveOnceResult, RetrieveOnceService

logger = get_logger(__name__)


@dataclass(slots=True)
class ParallelRetrievalResult:
    results: list[RetrieveOnceResult]
    tasks: list[ResearchTaskResult]
    success_count: int
    failed_count: int
    latency_ms: int
    degraded: bool


class ResearchParallelRetrievalService:
    def __init__(self, retrieve_once: RetrieveOnceService) -> None:
        self.retrieve_once = retrieve_once

    async def execute(
        self,
        *,
        state: ResearchState,
        queries: list[PlannedQuery],
        request: RetrieveRequest,
        knowledge_bases: list[Any],
        round: int,
    ) -> ParallelRetrievalResult:
        limit = MAX_INITIAL_QUERIES if round == 1 else MAX_SUPPLEMENT_QUERIES
        selected = queries[:limit]
        semaphore = asyncio.Semaphore(limit)
        started_at = perf_counter()
        logger.info(
            "RESEARCH_ROUND_START | research_id=%s | round=%s | query_count=%s",
            state.research_id,
            round,
            len(selected),
        )

        async def run_query(query: PlannedQuery) -> RetrieveOnceResult:
            async with semaphore:
                return await self.retrieve_once.execute(
                    tenant_id=state.tenant_id,
                    request=request,
                    knowledge_bases=knowledge_bases,
                    index_versions=dict(state.index_versions),
                    search_query=query.query,
                    query_id=query.query_id,
                    aspect_ids=list(query.aspect_ids),
                    round=round,
                )

        raw_results = await asyncio.gather(
            *(run_query(query) for query in selected),
            return_exceptions=True,
        )
        results: list[RetrieveOnceResult] = []
        tasks: list[ResearchTaskResult] = []
        for query, result in zip(selected, raw_results, strict=True):
            state.mark_visited(query.query)
            if isinstance(result, BaseException):
                error = type(result).__name__[:100]
                tasks.append(
                    ResearchTaskResult(
                        query_id=query.query_id,
                        query=query.query,
                        aspect_ids=query.aspect_ids,
                        round=round,
                        success=False,
                        latency_ms=0,
                        degraded=True,
                        error=error,
                    )
                )
                logger.warning(
                    "RESEARCH_TASK_FAILED | research_id=%s | round=%s | query_id=%s | "
                    "error=%s",
                    state.research_id,
                    round,
                    query.query_id,
                    error,
                )
                continue
            results.append(result)
            chunk_ids = [str(item.get("chunk_id") or "") for item in result.retrieved_chunks]
            tasks.append(
                ResearchTaskResult(
                    query_id=query.query_id,
                    query=query.query,
                    aspect_ids=query.aspect_ids,
                    round=round,
                    success=True,
                    latency_ms=result.latency_ms,
                    chunk_ids=[item for item in chunk_ids if item],
                    new_chunk_count=len({item for item in chunk_ids if item}),
                    degraded=result.degraded,
                    error=result.error,
                )
            )
            logger.info(
                "RESEARCH_TASK_SUCCESS | research_id=%s | round=%s | query_id=%s | "
                "candidate_count=%s | latency_ms=%s | degraded=%s",
                state.research_id,
                round,
                query.query_id,
                len(result.retrieved_chunks),
                result.latency_ms,
                result.degraded,
            )

        latency_ms = int((perf_counter() - started_at) * 1000)
        failed_count = len(selected) - len(results)
        degraded = failed_count > 0 or any(result.degraded for result in results)
        logger.info(
            "RESEARCH_ROUND_COMPLETE | research_id=%s | round=%s | success_count=%s | "
            "failed_count=%s | latency_ms=%s | degraded=%s",
            state.research_id,
            round,
            len(results),
            failed_count,
            latency_ms,
            degraded,
        )
        return ParallelRetrievalResult(
            results=results,
            tasks=tasks,
            success_count=len(results),
            failed_count=failed_count,
            latency_ms=latency_ms,
            degraded=degraded,
        )
