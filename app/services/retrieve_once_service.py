from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.core.index_versions import LEGACY_INDEX_VERSION
from app.schemas.rag import (
    RetrieveContextMetadata,
    RetrieveGraphInjectionMetadata,
    RetrieveRequest,
    RetrieveRerankMetadata,
)
from app.services.retrieval_candidate_filter import filter_retrieval_candidates


class RetrieveOnceBackend(Protocol):
    graph_candidate_expansion_service: Any
    context_expansion_service: Any

    async def _retrieve_multi_kb(self, **kwargs: Any) -> tuple[list[dict], dict]: ...

    def _resolve_empty_reason(
        self,
        tenant_id: str,
        kb_ids: list[str],
        chunks: list[dict],
        index_versions: dict[str, str] | None = None,
    ) -> str | None: ...

    async def _maybe_rerank(
        self,
        *,
        request: RetrieveRequest,
        rerank_query: str,
        chunks: list[dict],
        top_k: int,
    ) -> tuple[list[dict], RetrieveRerankMetadata]: ...

    def _get_retrieval_mode(self, request: RetrieveRequest) -> str: ...

    def _get_retrieval_candidate_k(self, request: RetrieveRequest, top_k: int) -> int: ...

@dataclass(slots=True)
class RetrieveOnceResult:
    query_id: str
    query: str
    aspect_ids: list[str]
    round: int
    candidate_snapshot: list[dict]
    retrieved_chunks: list[dict]
    retrieval_metadata: dict
    candidate_filter_metadata: dict
    graph_injection_metadata: RetrieveGraphInjectionMetadata
    context_metadata: RetrieveContextMetadata
    rerank_metadata: RetrieveRerankMetadata
    latency_ms: int
    degraded: bool = False
    error: str | None = None


class RetrieveOnceService:
    """Runs one deterministic retrieval pass without policy, quota, logs, or evidence LLMs."""

    def __init__(self, backend: RetrieveOnceBackend) -> None:
        self.backend = backend

    async def execute(
        self,
        *,
        tenant_id: str,
        request: RetrieveRequest,
        knowledge_bases: list[Any],
        index_versions: dict[str, str],
        search_query: str,
        query_id: str = "single",
        aspect_ids: list[str] | None = None,
        round: int = 1,
        expand_context: bool | None = None,
        graph_injection: bool | None = None,
    ) -> RetrieveOnceResult:
        started_at = perf_counter()
        kb_ids = request.kb_ids or []
        retrieval_mode = self.backend._get_retrieval_mode(request)
        top_k = request.top_k or settings.top_k
        retrieval_candidate_k = self.backend._get_retrieval_candidate_k(request, top_k)
        try:
            chunks, retrieval_metadata = await self.backend._retrieve_multi_kb(
                tenant_id=tenant_id,
                request=request,
                knowledge_bases=knowledge_bases,
                search_query=search_query,
                mode=retrieval_mode,
                top_k=retrieval_candidate_k,
                index_versions=index_versions,
            )
            empty_reason = self.backend._resolve_empty_reason(
                tenant_id,
                kb_ids,
                chunks,
                index_versions=index_versions,
            )
            if empty_reason:
                retrieval_metadata = {**retrieval_metadata, "empty_reason": empty_reason}
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                ErrorCode.RETRIEVAL_FAILED,
                internal_msg=f"retrieval failed: {exc}",
                context={"tenant_id": tenant_id, "kb_ids": kb_ids},
            ) from exc

        chunks, candidate_filter_metadata = filter_retrieval_candidates(chunks)
        should_inject_graph = graph_injection
        if should_inject_graph is None:
            should_inject_graph = any(
                version != LEGACY_INDEX_VERSION for version in index_versions.values()
            )
        if should_inject_graph:
            graph_expansion = (
                await self.backend.graph_candidate_expansion_service.expand_candidates(
                    tenant_id=tenant_id,
                    kb_ids=kb_ids,
                    seeds=chunks,
                )
            )
            chunks = graph_expansion.candidates
            graph_metadata = graph_expansion.metadata
        else:
            graph_metadata = RetrieveGraphInjectionMetadata(
                executed=False,
                candidate_count=len(chunks),
            )

        candidate_snapshot = [dict(chunk) for chunk in chunks]
        query_options = request.query_options
        rerank_query = (
            search_query
            if query_options and query_options.enabled and query_options.strategy == "rewrite"
            else request.query
        )
        chunks, rerank_metadata = await self.backend._maybe_rerank(
            request=request,
            rerank_query=rerank_query,
            chunks=chunks,
            top_k=min(top_k, len(chunks)) if chunks else top_k,
        )

        if expand_context is False:
            context_metadata = RetrieveContextMetadata(
                executed=False,
                anchor_count=len(chunks),
            )
        else:
            expansion = await self.backend.context_expansion_service.expand(
                tenant_id=tenant_id,
                kb_ids=kb_ids,
                anchors=chunks,
            )
            chunks = expansion.anchors
            context_metadata = expansion.metadata

        degraded = bool(
            retrieval_metadata.get("degraded")
            or graph_metadata.degraded
            or rerank_metadata.degraded
            or context_metadata.degraded
        )
        return RetrieveOnceResult(
            query_id=query_id,
            query=search_query,
            aspect_ids=list(aspect_ids or []),
            round=round,
            candidate_snapshot=candidate_snapshot,
            retrieved_chunks=chunks,
            retrieval_metadata=retrieval_metadata,
            candidate_filter_metadata=candidate_filter_metadata,
            graph_injection_metadata=graph_metadata,
            context_metadata=context_metadata,
            rerank_metadata=rerank_metadata,
            latency_ms=int((perf_counter() - started_at) * 1000),
            degraded=degraded,
        )
