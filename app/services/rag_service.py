import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.core.http_clients import ExternalHttpClients
from app.core.index_versions import DEFAULT_INDEX_VERSION
from app.core.logging import get_logger
from app.models.knowledge_base import KnowledgeBase
from app.observability.langfuse_client import RetrieveObservability
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.keyword_search.base import KeywordSearchProvider
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.query.base import KnowledgeBaseQueryContext, QueryProcessResult
from app.providers.query.pipeline import QueryProcessorPipeline
from app.providers.rerank.base import RerankProvider
from app.providers.rerank.noop import NoopRerankProvider
from app.providers.rerank.qwen3 import Qwen3RerankProvider
from app.providers.vectorstores.pgvector import PgVectorStore
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.index_version_repository import IndexVersionRepository
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.repositories.tenant_repository import TenantRepository
from app.schemas.rag import (
    EvidenceOptions,
    QueryProcessingMetadata,
    RetrieveData,
    RetrievedChunkData,
    RetrieveMetadata,
    RetrieveRequest,
    RetrieveRerankMetadata,
)
from app.services.context_expansion_service import ContextExpansionService
from app.services.evidence_orchestration_service import EvidenceOrchestrationService
from app.services.graph_candidate_expansion_service import GraphCandidateExpansionService
from app.services.hybrid_search_service import HybridSearchService
from app.services.rate_limit_service import get_rate_limit_service
from app.services.retrieve_once_service import RetrieveOnceService
from app.tenant.plan_resolver import PlanResolver
from app.tenant.retrieve_policy import resolve_retrieve_request
from app.tenant.retrieve_presets import FIXED_RRF_K
from app.utils.id_generator import generate_uuid

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalBranchResult:
    source: Literal["vector", "bm25"]
    chunks: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)
    error: Exception | None = None


class RagService:
    def __init__(
        self,
        db: Session,
        *,
        http_clients: ExternalHttpClients | None = None,
    ) -> None:
        self.db = db
        self.http_clients = http_clients
        self.knowledge_base_repository = KnowledgeBaseRepository(db)
        self.chunk_repository = ChunkRepository(db)
        self.index_version_repository = IndexVersionRepository(db)
        model_client = http_clients.model if http_clients else None
        self.embedding_provider = OpenAICompatibleEmbeddingProvider(client=model_client)
        self.vector_store = PgVectorStore(db)
        self.hybrid_search_service = HybridSearchService()
        self.query_processor = QueryProcessorPipeline(
            llm_provider=OpenAICompatibleLLMProvider(client=model_client)
        )
        self.context_expansion_service = ContextExpansionService(db)
        self.evidence_orchestration_service = EvidenceOrchestrationService(
            db,
            llm_provider=OpenAICompatibleLLMProvider(client=model_client),
        )
        self.graph_candidate_expansion_service = GraphCandidateExpansionService(db)
        self.retrieve_once_service = RetrieveOnceService(self)

    async def retrieve(
        self,
        tenant_id: str,
        request: RetrieveRequest,
        *,
        index_version_overrides: dict[str, str] | None = None,
        expand_context: bool | None = None,
        graph_injection: bool | None = None,
    ) -> RetrieveData:
        tenant = TenantRepository(self.db).get_by_id(tenant_id)
        if tenant is None:
            raise AppError(
                ErrorCode.NOT_FOUND,
                msg="租户不存在",
                context={"tenant_id": tenant_id},
            )

        plan_context = PlanResolver.resolve_plan(tenant)
        resolved_request, tenant_policy = resolve_retrieve_request(request, plan_context)
        kb_ids = resolved_request.kb_ids or []
        knowledge_bases = self._load_knowledge_bases(tenant_id, kb_ids)
        index_versions = self._resolve_index_versions(
            tenant_id,
            knowledge_bases,
            requested_version=resolved_request.index_version,
            index_version_overrides=index_version_overrides,
        )

        get_rate_limit_service().check_retrieve(
            tenant_id,
            qps_limit=plan_context.limits.retrieve_qps,
            daily_limit=plan_context.limits.daily_retrieve_limit,
        )

        retrieval_mode = self._get_retrieval_mode(resolved_request)
        top_k = resolved_request.top_k or settings.top_k
        logger.info(
            "BUSINESS_NODE | rag_retrieve_start | tenant_id=%s | kb_ids=%s | user_id=%s | "
            "plan=%s | profile=%s | mode=%s | top_k=%s | retrieval_options=%s | "
            "rerank_options=%s | query_options=%s | index_versions=%s",
            tenant_id,
            kb_ids,
            resolved_request.user_id,
            tenant_policy.plan,
            tenant_policy.retrieve_profile,
            retrieval_mode,
            top_k,
            resolved_request.retrieval_options.model_dump()
            if resolved_request.retrieval_options
            else None,
            resolved_request.rerank_options.model_dump()
            if resolved_request.rerank_options
            else None,
            resolved_request.query_options.model_dump() if resolved_request.query_options else None,
            index_versions,
        )

        log_id = generate_uuid()
        observability = RetrieveObservability(log_id)
        if resolved_request.observability_enabled is not False:
            observability.start(
                tenant_id=tenant_id,
                kb_ids=kb_ids,
                user_id=resolved_request.user_id,
                raw_query=resolved_request.query,
                profile=tenant_policy.retrieve_profile,
                plan=tenant_policy.plan,
                mode=retrieval_mode,
                top_k=top_k,
            )

        query_result = await self.query_processor.process(
            resolved_request.query,
            kb=self._build_query_context(knowledge_bases),
            query_options=resolved_request.query_options,
        )
        search_query = query_result.search_query
        query_processing = self._build_query_processing_metadata(query_result)
        observability.record_query_processing(
            query_processing.model_dump() if query_processing else None
        )

        once = await self.retrieve_once_service.execute(
            tenant_id=tenant_id,
            request=resolved_request,
            knowledge_bases=knowledge_bases,
            index_versions=index_versions,
            search_query=search_query,
            expand_context=expand_context,
            graph_injection=graph_injection,
        )
        chunks = once.retrieved_chunks
        candidate_snapshot = once.candidate_snapshot
        retrieval_metadata = once.retrieval_metadata
        candidate_filter_metadata = once.candidate_filter_metadata
        graph_injection_metadata = once.graph_injection_metadata
        context_metadata = once.context_metadata
        rerank_metadata = once.rerank_metadata

        evidence_result = await self.evidence_orchestration_service.orchestrate(
            query=resolved_request.query,
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            index_versions=index_versions,
            candidates=candidate_snapshot,
            retrieved_chunks=chunks,
            options=resolved_request.evidence_options or EvidenceOptions(enabled=False),
            rerank_enabled=rerank_metadata.enabled,
        )
        latency_ms = once.latency_ms
        observability.record_retrieval(retrieval_metadata, latency_ms=latency_ms)
        observability.record_rerank(rerank_metadata.model_dump())
        observability.finish(chunks, latency_ms=latency_ms)

        primary_kb_id = kb_ids[0]
        get_rate_limit_service().record_retrieve_success(tenant_id)

        logger.info(
            "BUSINESS_NODE | rag_retrieve_success | kb_ids=%s | mode=%s | returned_count=%s | "
            "latency_ms=%s | retrieval=%s | rerank=%s | query_processing=%s | tenant_policy=%s",
            kb_ids,
            retrieval_mode,
            len(chunks),
            latency_ms,
            retrieval_metadata,
            rerank_metadata.model_dump(),
            query_processing.model_dump() if query_processing else None,
            tenant_policy.model_dump(),
        )
        return RetrieveData(
            query=resolved_request.query,
            kb_id=primary_kb_id,
            kb_ids=kb_ids,
            retrieved_chunks=[RetrievedChunkData(**chunk) for chunk in chunks],
            evidence_pack=evidence_result.pack,
            metadata=RetrieveMetadata(
                log_id=log_id,
                trace_id=observability.trace_id,
                top_k=top_k,
                vector_store=self.vector_store.name,
                latency_ms=latency_ms,
                retrieval=retrieval_metadata,
                candidate_filter=candidate_filter_metadata,
                graph_injection=graph_injection_metadata,
                context=context_metadata,
                evidence=evidence_result.metadata,
                rerank=rerank_metadata,
                query_processing=query_processing,
                tenant_policy=tenant_policy,
            ),
        )

    def _load_knowledge_bases(self, tenant_id: str, kb_ids: list[str]) -> list[KnowledgeBase]:
        knowledge_bases: list[KnowledgeBase] = []
        for kb_id in kb_ids:
            knowledge_base = self.knowledge_base_repository.get_by_id_and_tenant(kb_id, tenant_id)
            if knowledge_base is None:
                raise AppError(
                    ErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
                    context={
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "missing_kb_id": kb_id,
                    },
                )
            knowledge_bases.append(knowledge_base)
        return knowledge_bases

    def _resolve_index_versions(
        self,
        tenant_id: str,
        knowledge_bases: list[KnowledgeBase],
        *,
        requested_version: str | None,
        index_version_overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for knowledge_base in knowledge_bases:
            if index_version_overrides and knowledge_base.id in index_version_overrides:
                version = index_version_overrides[knowledge_base.id]
            elif requested_version:
                version = requested_version
            else:
                version = knowledge_base.active_index_version
            item = self.index_version_repository.get(tenant_id, knowledge_base.id, version)
            allowed_statuses = {"active", "ready", "archived"}
            if item is None or item.status not in allowed_statuses:
                raise AppError(
                    ErrorCode.NOT_FOUND,
                    msg="索引版本不存在或不可查询",
                    context={
                        "tenant_id": tenant_id,
                        "kb_id": knowledge_base.id,
                        "index_version": version,
                        "status": item.status if item else None,
                    },
                )
            resolved[knowledge_base.id] = version
        return resolved

    @staticmethod
    def _build_query_context(knowledge_bases: list[KnowledgeBase]) -> KnowledgeBaseQueryContext:
        primary = knowledge_bases[0]
        if len(knowledge_bases) == 1:
            return KnowledgeBaseQueryContext(
                name=primary.name,
                description=primary.description,
                settings=primary.settings or {},
            )

        names = "、".join(kb.name for kb in knowledge_bases)
        descriptions = [kb.description for kb in knowledge_bases if kb.description]
        merged_description = "\n".join(descriptions) if descriptions else f"联合检索知识库：{names}"
        return KnowledgeBaseQueryContext(
            name=names,
            description=merged_description,
            settings=primary.settings or {},
        )

    async def _retrieve_multi_kb(
        self,
        *,
        tenant_id: str,
        request: RetrieveRequest,
        knowledge_bases: list[KnowledgeBase],
        search_query: str,
        mode: str,
        top_k: int,
        index_versions: dict[str, str] | None = None,
    ) -> tuple[list[dict], dict]:
        effective_versions = index_versions or {
            kb.id: getattr(kb, "active_index_version", DEFAULT_INDEX_VERSION)
            for kb in knowledge_bases
        }
        vector_top_k = self._get_vector_top_k(request, top_k)
        bm25_top_k = self._get_bm25_top_k(request, top_k)
        rrf_k = self._get_rrf_k(request)
        hybrid_started_at = perf_counter()

        if mode == "vector":
            vector_result = await self._run_vector_scope_branch(
                tenant_id=tenant_id,
                index_versions=effective_versions,
                search_query=search_query,
                top_k=vector_top_k,
            )
            if vector_result.error is not None:
                self._raise_retrieval_failed(
                    tenant_id=tenant_id,
                    index_versions=effective_versions,
                    errors=[vector_result.error],
                )
            chunks = self._mark_vector_chunks(vector_result.chunks)[:top_k]
            return self._tag_scope_chunks(chunks, knowledge_bases), self._scope_metadata(
                mode=mode,
                chunks=chunks,
                index_versions=effective_versions,
                vector_top_k=vector_top_k,
                bm25_top_k=None,
                vector_result=vector_result,
                bm25_result=None,
                rrf_k=None,
                hybrid_started_at=hybrid_started_at,
            )

        if mode == "bm25":
            bm25_result = await self._run_bm25_scope_branch(
                tenant_id=tenant_id,
                index_versions=effective_versions,
                search_query=search_query,
                top_k=bm25_top_k,
            )
            if bm25_result.error is not None:
                self._raise_retrieval_failed(
                    tenant_id=tenant_id,
                    index_versions=effective_versions,
                    errors=[bm25_result.error],
                )
            chunks = self._mark_bm25_chunks(bm25_result.chunks)[:top_k]
            return self._tag_scope_chunks(chunks, knowledge_bases), self._scope_metadata(
                mode=mode,
                chunks=chunks,
                index_versions=effective_versions,
                vector_top_k=None,
                bm25_top_k=bm25_top_k,
                vector_result=None,
                bm25_result=bm25_result,
                rrf_k=None,
                hybrid_started_at=hybrid_started_at,
            )

        if mode != "hybrid":
            raise AppError(
                ErrorCode.PARAM_ERROR,
                internal_msg=f"unsupported retrieval mode: {mode}",
                context={"mode": mode},
            )

        vector_result, bm25_result = await asyncio.gather(
            self._run_vector_scope_branch(
                tenant_id=tenant_id,
                index_versions=effective_versions,
                search_query=search_query,
                top_k=vector_top_k,
            ),
            self._run_bm25_scope_branch(
                tenant_id=tenant_id,
                index_versions=effective_versions,
                search_query=search_query,
                top_k=bm25_top_k,
            ),
        )
        if vector_result.error is not None and bm25_result.error is not None:
            self._raise_retrieval_failed(
                tenant_id=tenant_id,
                index_versions=effective_versions,
                errors=[vector_result.error, bm25_result.error],
            )
        if vector_result.error is not None:
            chunks = self._mark_bm25_chunks(bm25_result.chunks)[:top_k]
        elif bm25_result.error is not None:
            chunks = self._mark_vector_chunks(vector_result.chunks)[:top_k]
        else:
            chunks = self.hybrid_search_service.fuse_by_rrf(
                vector_chunks=vector_result.chunks,
                bm25_chunks=bm25_result.chunks,
                rrf_k=rrf_k,
                top_n=top_k,
            )
        tagged = self._tag_scope_chunks(chunks, knowledge_bases)
        return tagged, self._scope_metadata(
            mode=mode,
            chunks=tagged,
            index_versions=effective_versions,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            vector_result=vector_result,
            bm25_result=bm25_result,
            rrf_k=rrf_k,
            hybrid_started_at=hybrid_started_at,
        )

    @staticmethod
    def _get_retrieval_candidate_k(request: RetrieveRequest, top_k: int) -> int:
        options = request.retrieval_options
        configured = [top_k]
        if options and options.vector_top_k is not None:
            configured.append(options.vector_top_k)
        if options and options.bm25_top_k is not None:
            configured.append(options.bm25_top_k)
        return max(configured)

    @staticmethod
    def _build_query_processing_metadata(
        query_result: QueryProcessResult,
    ) -> QueryProcessingMetadata | None:
        if not query_result.should_expose_metadata():
            return None

        return QueryProcessingMetadata(
            enabled=query_result.rewrite_attempted,
            strategy=query_result.strategy,
            raw_query=query_result.raw_query,
            effective_query=query_result.effective_query,
            search_query=query_result.search_query,
            latency_ms=query_result.latency_ms,
            synonym_expansions=query_result.synonym_expansions,
            synonym_enabled=query_result.synonym_enabled,
            synonym_applied=query_result.synonym_applied,
            degraded=query_result.degraded,
            degraded_reason=query_result.degraded_reason,
        )

    async def _run_vector_scope_branch(
        self,
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        search_query: str,
        top_k: int,
    ) -> RetrievalBranchResult:
        started_at = perf_counter()
        details = {
            "embedding_request_count": 1,
            "vector_request_count": 0,
            "embedding_latency_ms": 0.0,
            "vector_store_latency_ms": 0.0,
        }
        try:
            embedding_started_at = perf_counter()
            query_vector = await self.embedding_provider.embed_query(search_query)
            details["embedding_latency_ms"] = round(
                (perf_counter() - embedding_started_at) * 1000,
                2,
            )
            vector_store_started_at = perf_counter()
            details["vector_request_count"] = 1
            chunks = await self.vector_store.similarity_search_scope(
                query_vector,
                tenant_id=tenant_id,
                index_versions=index_versions,
                top_k=top_k,
            )
            details["vector_store_latency_ms"] = round(
                (perf_counter() - vector_store_started_at) * 1000,
                2,
            )
            return RetrievalBranchResult(
                source="vector",
                chunks=chunks,
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                details=details,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "VECTOR_SCOPE_SEARCH_FAILED | tenant_id=%s | kb_ids=%s | "
                "error_type=%s | error=%s",
                tenant_id,
                sorted(index_versions),
                type(exc).__name__,
                str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return RetrievalBranchResult(
                source="vector",
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                details=details,
                error=exc,
            )

    async def _run_bm25_scope_branch(
        self,
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        search_query: str,
        top_k: int,
    ) -> RetrievalBranchResult:
        started_at = perf_counter()
        details = {"bm25_request_count": 0}
        try:
            provider = self._build_keyword_search_provider()
            details["bm25_request_count"] = 1
            chunks = await provider.keyword_search_scope(
                query=search_query,
                tenant_id=tenant_id,
                index_versions=index_versions,
                top_k=top_k,
            )
            return RetrievalBranchResult(
                source="bm25",
                chunks=chunks,
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                details=details,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "BM25_SCOPE_SEARCH_FAILED | tenant_id=%s | kb_ids=%s | "
                "error_type=%s | error=%s",
                tenant_id,
                sorted(index_versions),
                type(exc).__name__,
                str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return RetrievalBranchResult(
                source="bm25",
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                details=details,
                error=exc,
            )

    @staticmethod
    def _tag_scope_chunks(
        chunks: list[dict],
        knowledge_bases: list[KnowledgeBase],
    ) -> list[dict]:
        names = {str(kb.id): kb.name for kb in knowledge_bases}
        return [
            {
                **chunk,
                "kb_name": names.get(str(chunk.get("kb_id") or "")),
            }
            for chunk in chunks
        ]

    def _scope_metadata(
        self,
        *,
        mode: str,
        chunks: list[dict],
        index_versions: dict[str, str],
        vector_top_k: int | None,
        bm25_top_k: int | None,
        vector_result: RetrievalBranchResult | None,
        bm25_result: RetrievalBranchResult | None,
        rrf_k: int | None,
        hybrid_started_at: float,
    ) -> dict:
        degraded_sources = [
            result.source
            for result in (vector_result, bm25_result)
            if result is not None and result.error is not None
        ]
        source_distribution: dict[str, int] = {}
        for chunk in chunks:
            kb_id = str(chunk.get("kb_id") or "")
            source_distribution[kb_id] = source_distribution.get(kb_id, 0) + 1
        vector_details = vector_result.details if vector_result else {}
        bm25_details = bm25_result.details if bm25_result else {}
        hybrid_latency_ms = round((perf_counter() - hybrid_started_at) * 1000, 2)
        return {
            "mode": mode,
            "fusion": "rrf" if mode == "hybrid" else None,
            "fusion_scope": "global",
            "rrf_k": rrf_k,
            "vector_store": self.vector_store.name,
            "keyword_search": settings.keyword_search_provider if bm25_result else None,
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "vector_count": len(vector_result.chunks) if vector_result else 0,
            "bm25_count": len(bm25_result.chunks) if bm25_result else 0,
            "fused_count": len(chunks),
            "embedding_request_count": int(
                vector_details.get("embedding_request_count") or 0
            ),
            "vector_request_count": int(vector_details.get("vector_request_count") or 0),
            "bm25_request_count": int(bm25_details.get("bm25_request_count") or 0),
            "embedding_latency_ms": vector_details.get("embedding_latency_ms"),
            "vector_store_latency_ms": vector_details.get("vector_store_latency_ms"),
            "vector_latency_ms": vector_result.latency_ms if vector_result else None,
            "bm25_latency_ms": bm25_result.latency_ms if bm25_result else None,
            "hybrid_latency_ms": hybrid_latency_ms if mode == "hybrid" else None,
            "cost_ms": hybrid_latency_ms,
            "degraded": bool(degraded_sources),
            "degraded_reason": (
                "; ".join(f"{source} search failed" for source in degraded_sources)
                if degraded_sources
                else None
            ),
            "multi_kb": len(index_versions) > 1,
            "kb_count": len(index_versions),
            "kb_ids": sorted(index_versions),
            "index_versions": dict(sorted(index_versions.items())),
            "source_distribution": dict(sorted(source_distribution.items())),
            "empty_reason": None,
        }

    @staticmethod
    def _raise_retrieval_failed(
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        errors: list[Exception | None],
    ) -> None:
        first_error = next((error for error in errors if error is not None), None)
        raise AppError(
            ErrorCode.RETRIEVAL_FAILED,
            internal_msg="retrieval scope failed",
            context={
                "tenant_id": tenant_id,
                "kb_ids": sorted(index_versions),
            },
        ) from first_error

    def _count_indexed_chunks(
        self,
        tenant_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str] | None = None,
    ) -> int:
        return sum(
            self.chunk_repository.count_by_kb_id(
                tenant_id,
                kb_id,
                index_version=index_versions.get(kb_id) if index_versions else None,
            )
            for kb_id in kb_ids
        )

    def _resolve_empty_reason(
        self,
        tenant_id: str,
        kb_ids: list[str],
        chunks: list[dict],
        index_versions: dict[str, str] | None = None,
    ) -> str | None:
        if chunks:
            return None
        if self._count_indexed_chunks(tenant_id, kb_ids, index_versions=index_versions) == 0:
            return "no_indexed_chunks"
        return "no_chunks_matched"

    def _mark_vector_chunks(self, chunks: list[dict]) -> list[dict]:
        marked: list[dict] = []
        for rank, chunk in enumerate(chunks, start=1):
            marked.append(
                {
                    **chunk,
                    "vector_score": chunk.get("score"),
                    "bm25_score": None,
                    "vector_rank": rank,
                    "bm25_rank": None,
                    "retrieval_source": "vector",
                }
            )
        return marked

    def _mark_bm25_chunks(self, chunks: list[dict]) -> list[dict]:
        marked: list[dict] = []
        for rank, chunk in enumerate(chunks, start=1):
            marked.append(
                {
                    **chunk,
                    "score": chunk.get("bm25_score", 0.0),
                    "vector_score": None,
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "retrieval_source": "bm25",
                }
            )
        return marked

    async def _maybe_rerank(
        self,
        *,
        request: RetrieveRequest,
        rerank_query: str,
        chunks: list[dict],
        top_k: int,
    ) -> tuple[list[dict], RetrieveRerankMetadata]:
        enabled = (
            request.rerank_options.enabled
            if request.rerank_options and request.rerank_options.enabled is not None
            else settings.rerank_enabled
        )
        top_n = (
            request.rerank_options.top_n
            if request.rerank_options and request.rerank_options.top_n is not None
            else top_k
        )
        top_n = min(top_n, top_k, len(chunks)) if chunks else 0

        if not enabled:
            reranked = await NoopRerankProvider().rerank(
                query=rerank_query,
                chunks=chunks,
                top_n=min(top_k, len(chunks)),
            )
            return reranked, RetrieveRerankMetadata(
                enabled=False,
                provider="noop",
                top_n=None,
                candidate_count=len(chunks),
                returned_count=len(reranked),
                latency_ms=0,
            )

        rerank_provider = self._build_rerank_provider()
        candidate_count = min(len(chunks), rerank_provider.max_candidates)
        rerank_started_at = perf_counter()
        try:
            reranked = await rerank_provider.rerank(
                query=rerank_query,
                chunks=chunks,
                top_n=top_n,
            )
            return reranked, RetrieveRerankMetadata(
                enabled=True,
                provider=rerank_provider.name,
                model=rerank_provider.model,
                top_n=top_n,
                candidate_count=candidate_count,
                returned_count=len(reranked),
                latency_ms=int((perf_counter() - rerank_started_at) * 1000),
            )
        except Exception as exc:
            logger.warning(
                "BUSINESS_NODE | rerank_degraded | kb_ids=%s | candidate_count=%s | error_type=%s | error=%s",
                request.kb_ids,
                candidate_count,
                type(exc).__name__,
                str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            fallback_top_n = min(top_n, len(chunks))
            fallback_chunks = [{**chunk, "rerank_score": None} for chunk in chunks[:fallback_top_n]]
            return fallback_chunks, RetrieveRerankMetadata(
                enabled=True,
                provider=rerank_provider.name,
                model=rerank_provider.model,
                top_n=top_n,
                candidate_count=candidate_count,
                returned_count=len(fallback_chunks),
                latency_ms=int((perf_counter() - rerank_started_at) * 1000),
                degraded=True,
                error=str(exc),
            )

    def _build_rerank_provider(self) -> RerankProvider:
        return Qwen3RerankProvider(
            client=self.http_clients.model if self.http_clients else None
        )

    def _build_keyword_search_provider(self) -> KeywordSearchProvider:
        if settings.keyword_search_provider == "elasticsearch":
            return ElasticsearchKeywordSearchProvider(
                client=self.http_clients.elasticsearch if self.http_clients else None
            )
        raise ValueError(f"unsupported keyword search provider: {settings.keyword_search_provider}")

    def _get_retrieval_mode(self, request: RetrieveRequest) -> str:
        if request.retrieval_options and request.retrieval_options.mode:
            return request.retrieval_options.mode
        return settings.retrieval_mode

    def _get_vector_top_k(self, request: RetrieveRequest, fallback: int) -> int:
        if request.retrieval_options and request.retrieval_options.vector_top_k:
            return request.retrieval_options.vector_top_k
        return settings.hybrid_vector_top_k or fallback

    def _get_bm25_top_k(self, request: RetrieveRequest, fallback: int) -> int:
        if request.retrieval_options and request.retrieval_options.bm25_top_k:
            return request.retrieval_options.bm25_top_k
        if self._get_retrieval_mode(request) == "bm25":
            return fallback
        return settings.hybrid_bm25_top_k or fallback

    def _get_rrf_k(self, request: RetrieveRequest) -> int:
        if request.retrieval_options and request.retrieval_options.rrf_k:
            return request.retrieval_options.rrf_k
        return FIXED_RRF_K
