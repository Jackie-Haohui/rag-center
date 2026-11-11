import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.core.index_versions import DEFAULT_INDEX_VERSION, LEGACY_INDEX_VERSION
from app.core.logging import get_logger, safe_json
from app.providers.keyword_search.base import KeywordSearchProvider

logger = get_logger(__name__)


class ElasticsearchKeywordSearchProvider(KeywordSearchProvider):
    name = "elasticsearch"

    def __init__(
        self,
        *,
        base_url: str = settings.elasticsearch_url,
        index_name: str = settings.elasticsearch_index,
        timeout: float = settings.elasticsearch_timeout_seconds,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.index_name = index_name
        self.timeout = timeout
        self.client = client

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            yield client

    async def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        started_at = perf_counter()
        await self.ensure_index()
        bulk_lines: list[str] = []
        now = datetime.now(UTC).isoformat()

        for chunk in chunks:
            doc_id = chunk["chunk_id"]
            source = {
                "tenant_id": chunk["tenant_id"],
                "kb_id": chunk["kb_id"],
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "retrieval_text": chunk.get("retrieval_text"),
                "index_version": chunk.get("index_version", DEFAULT_INDEX_VERSION),
                "section_id": chunk.get("section_id"),
                "parent_section_id": chunk.get("parent_section_id"),
                "order_index": chunk.get("order_index"),
                "metadata": chunk.get("metadata", {}),
                "created_at": now,
            }
            bulk_lines.append(
                json.dumps(
                    {"index": {"_index": self.index_name, "_id": doc_id}},
                    ensure_ascii=False,
                )
            )
            bulk_lines.append(json.dumps(source, ensure_ascii=False, default=str))

        payload = "\n".join(bulk_lines) + "\n"
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/_bulk",
                    content=payload,
                    headers={"Content-Type": "application/x-ndjson"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            error_body = _response_text(exc)
            logger.error(
                "KEYWORD_INDEX_FAILED | provider=%s | index=%s | chunk_count=%s | "
                "error=%s | response_body=%s",
                self.name,
                self.index_name,
                len(chunks),
                str(exc),
                error_body,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise AppError(
                ErrorCode.KEYWORD_SEARCH_ERROR,
                internal_msg=f"elasticsearch bulk index failed: {exc}; response_body={error_body}",
                context={"index": self.index_name, "chunk_count": len(chunks)},
            ) from exc

        if body.get("errors"):
            logger.error(
                "KEYWORD_INDEX_FAILED | provider=%s | index=%s | chunk_count=%s | response=%s",
                self.name,
                self.index_name,
                len(chunks),
                safe_json(body),
            )
            raise AppError(
                ErrorCode.KEYWORD_SEARCH_ERROR,
                internal_msg="elasticsearch bulk index returned item errors",
                context={"index": self.index_name, "chunk_count": len(chunks), "response": body},
            )

        cost_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "KEYWORD_INDEX_SUCCESS | provider=%s | index=%s | chunk_count=%s | cost_ms=%.2f",
            self.name,
            self.index_name,
            len(chunks),
            cost_ms,
        )

    async def keyword_search(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_id: str,
        index_version: str | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        return await self.keyword_search_scope(
            query=query,
            tenant_id=tenant_id,
            index_versions={kb_id: index_version or DEFAULT_INDEX_VERSION},
            top_k=top_k,
        )

    async def keyword_search_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        index_versions: dict[str, str],
        top_k: int = 20,
    ) -> list[dict]:
        if not index_versions:
            return []
        started_at = perf_counter()
        payload = _build_scope_search_payload(
            query=query,
            tenant_id=tenant_id,
            index_versions=index_versions,
            top_k=top_k,
        )

        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/{self.index_name}/_search",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            error_body = _response_text(exc)
            logger.error(
                "BM25_SEARCH_FAILED | provider=%s | index=%s | tenant_id=%s | kb_ids=%s | "
                "top_k=%s | error=%s | response_body=%s",
                self.name,
                self.index_name,
                tenant_id,
                sorted(index_versions),
                top_k,
                str(exc),
                error_body,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise AppError(
                ErrorCode.KEYWORD_SEARCH_ERROR,
                internal_msg=f"elasticsearch bm25 search failed: {exc}; response_body={error_body}",
                context={
                    "index": self.index_name,
                    "tenant_id": tenant_id,
                    "kb_ids": sorted(index_versions),
                },
            ) from exc

        results: list[dict] = []
        for hit in body.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append(
                {
                    "kb_id": source.get("kb_id"),
                    "document_id": source.get("document_id"),
                    "chunk_id": source.get("chunk_id"),
                    "title": source.get("title", ""),
                    "content": source.get("content", ""),
                    "retrieval_text": source.get("retrieval_text"),
                    "index_version": source.get("index_version", DEFAULT_INDEX_VERSION),
                    "section_id": source.get("section_id"),
                    "parent_section_id": source.get("parent_section_id"),
                    "order_index": source.get("order_index"),
                    "bm25_score": float(hit.get("_score") or 0.0),
                    "metadata": source.get("metadata", {}),
                }
            )

        cost_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "BM25_SEARCH_SUCCESS | provider=%s | index=%s | tenant_id=%s | kb_ids=%s | "
            "top_k=%s | result_count=%s | cost_ms=%.2f",
            self.name,
            self.index_name,
            tenant_id,
            sorted(index_versions),
            top_k,
            len(results),
            cost_ms,
        )
        return results

    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if index_version is not None:
            filters.append({"term": {"index_version": index_version}})
        payload = {"query": {"bool": {"filter": filters}}}
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/{self.index_name}/_delete_by_query",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            error_body = _response_text(exc)
            logger.error(
                "KEYWORD_DELETE_FAILED | provider=%s | index=%s | document_id=%s | "
                "error=%s | response_body=%s",
                self.name,
                self.index_name,
                document_id,
                str(exc),
                error_body,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise AppError(
                ErrorCode.KEYWORD_SEARCH_ERROR,
                internal_msg=f"elasticsearch delete failed: {exc}; response_body={error_body}",
                context={"index": self.index_name, "document_id": document_id},
            ) from exc

    async def delete_by_kb_version(self, *, tenant_id: str, kb_id: str, index_version: str) -> None:
        payload = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tenant_id": tenant_id}},
                        {"term": {"kb_id": kb_id}},
                        {"term": {"index_version": index_version}},
                    ]
                }
            }
        }
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/{self.index_name}/_delete_by_query?refresh=true&conflicts=proceed",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

    async def ensure_index(self) -> None:
        try:
            async with self._client() as client:
                response = await client.head(
                    f"{self.base_url}/{self.index_name}", timeout=self.timeout
                )
                if response.status_code == 200:
                    await self._ensure_mapping(client)
                    return
                if response.status_code != 404:
                    response.raise_for_status()

                create_response = await client.put(
                    f"{self.base_url}/{self.index_name}",
                    json=self._index_body(),
                    timeout=self.timeout,
                )
                create_response.raise_for_status()
        except httpx.HTTPError as exc:
            error_body = _response_text(exc)
            logger.error(
                "KEYWORD_INDEX_INIT_FAILED | provider=%s | index=%s | error=%s | response_body=%s",
                self.name,
                self.index_name,
                str(exc),
                error_body,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise AppError(
                ErrorCode.KEYWORD_SEARCH_ERROR,
                internal_msg=f"elasticsearch index init failed: {exc}; response_body={error_body}",
                context={"index": self.index_name},
            ) from exc

    def _index_body(self) -> dict[str, Any]:
        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "ik_smart",
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "tenant_id": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "index_version": {"type": "keyword"},
                    "section_id": {"type": "keyword"},
                    "parent_section_id": {"type": "keyword"},
                    "order_index": {"type": "integer"},
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "retrieval_text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "metadata": {"type": "object", "enabled": False},
                    "created_at": {"type": "date"},
                }
            },
        }

    async def _ensure_mapping(self, client: httpx.AsyncClient) -> None:
        response = await client.put(
            f"{self.base_url}/{self.index_name}/_mapping",
            json={
                "properties": {
                    "index_version": {"type": "keyword"},
                    "retrieval_text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "section_id": {"type": "keyword"},
                    "parent_section_id": {"type": "keyword"},
                    "order_index": {"type": "integer"},
                }
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

    async def backfill_v1_index_version(self) -> dict[str, int]:
        await self.ensure_index()
        payload = {
            "script": {
                "source": "if (ctx._source.index_version == null) { ctx._source.index_version = 'v1' }",
                "lang": "painless",
            },
            "query": {"bool": {"must_not": {"exists": {"field": "index_version"}}}},
        }
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/{self.index_name}/_update_by_query?refresh=true&conflicts=proceed",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        return {
            "total": int(body.get("total") or 0),
            "updated": int(body.get("updated") or 0),
            "version_conflicts": int(body.get("version_conflicts") or 0),
        }

    async def count_by_version(
        self, *, tenant_id: str, kb_id: str, index_version: str | None
    ) -> int:
        filters: list[dict[str, Any]] = [
            {"term": {"tenant_id": tenant_id}},
            {"term": {"kb_id": kb_id}},
        ]
        if index_version is None:
            filters.append({"bool": {"must_not": {"exists": {"field": "index_version"}}}})
        else:
            filters.append({"term": {"index_version": index_version}})
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/{self.index_name}/_count",
                json={"query": {"bool": {"filter": filters}}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return int(response.json().get("count") or 0)


def _build_search_payload(
    *,
    query: str,
    filters: list[dict[str, Any]],
    index_version: str | None,
    top_k: int,
) -> dict[str, Any]:
    is_legacy_index = not index_version or index_version == LEGACY_INDEX_VERSION
    fields = (
        ["metadata.heading_path^2", "content^2", "title"]
        if is_legacy_index
        else ["retrieval_text"]
    )
    multi_match: dict[str, Any] = {
        "query": query,
        "fields": fields,
    }
    if is_legacy_index:
        # Chinese questions contain many function words. Requiring half of
        # the analyzed terms keeps exact codes and business phrases while
        # preventing a single common word from producing a noisy candidate.
        multi_match.update(
            {
                "type": "cross_fields",
                "minimum_should_match": "50%",
            }
        )
    return {
        "query": {
            "bool": {
                "filter": filters,
                "must": {"multi_match": multi_match},
            }
        },
        "size": top_k,
        "sort": [
            {"_score": {"order": "desc"}},
            {"kb_id": {"order": "asc"}},
            {"chunk_id": {"order": "asc"}},
        ],
    }


def _build_scope_search_payload(
    *,
    query: str,
    tenant_id: str,
    index_versions: dict[str, str],
    top_k: int,
) -> dict[str, Any]:
    scoped_queries: list[dict[str, Any]] = []
    for kb_id, index_version in sorted(index_versions.items()):
        is_legacy_index = index_version == LEGACY_INDEX_VERSION
        fields = (
            ["metadata.heading_path^2", "content^2", "title"]
            if is_legacy_index
            else ["retrieval_text"]
        )
        multi_match: dict[str, Any] = {"query": query, "fields": fields}
        if is_legacy_index:
            multi_match.update(
                {
                    "type": "cross_fields",
                    "minimum_should_match": "50%",
                }
            )
        scoped_queries.append(
            {
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"index_version": index_version}},
                    ],
                    "must": [{"multi_match": multi_match}],
                }
            }
        )
    return {
        "query": {
            "bool": {
                "filter": [{"term": {"tenant_id": tenant_id}}],
                "should": scoped_queries,
                "minimum_should_match": 1,
            }
        },
        "size": top_k,
        "sort": [
            {"_score": {"order": "desc"}},
            {"kb_id": {"order": "asc"}},
            {"chunk_id": {"order": "asc"}},
        ],
    }


def _response_text(exc: httpx.HTTPError) -> str | None:
    response = getattr(exc, "response", None)
    return response.text if response is not None else None
