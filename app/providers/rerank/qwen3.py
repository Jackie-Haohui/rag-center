from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.providers.rerank.base import RerankProvider, RerankProviderError

logger = get_logger(__name__)

QWEN3_RERANK_INSTRUCT = (
    "根据用户问题对候选片段进行严格相关性排序。"
    "只提高能够直接回答问题、提供明确规则、条件、时间、金额或结论的片段。"
    "仅主题相似、只提供背景信息、需要推测才能关联的片段应降低排名。"
    "对于包含多个子问题的查询，优先覆盖所有子问题。"
    "重复或信息高度重叠的片段应降低排名。"
)
QWEN3_RERANK_MAX_CANDIDATES = 50
QWEN3_RERANK_CHUNK_MAX_CHARS = 1024


class Qwen3RerankProvider(RerankProvider):
    name = "qwen3"

    def __init__(
        self,
        *,
        base_url: str = settings.rerank_base_url,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = settings.rerank_timeout_seconds,
        instruct: str = QWEN3_RERANK_INSTRUCT,
        max_candidates: int = QWEN3_RERANK_MAX_CANDIDATES,
        chunk_max_chars: int = QWEN3_RERANK_CHUNK_MAX_CHARS,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key if api_key is not None else settings.rerank_api_key or settings.model_api_key
        )
        self.model = model or settings.rerank_model
        self.timeout_seconds = timeout_seconds
        self.instruct = instruct
        self.max_candidates = max_candidates
        self.chunk_max_chars = chunk_max_chars
        self.transport = transport
        self.client = client
        self.last_usage: dict[str, Any] = {}

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            yield client

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        candidates = chunks[: self.max_candidates]
        if not candidates or top_n <= 0:
            return []

        if not self.base_url:
            raise RerankProviderError("qwen3 rerank base URL is not configured")
        if not self.api_key or self.api_key.startswith("your-"):
            raise RerankProviderError("qwen3 rerank API key is not configured")

        effective_top_n = min(top_n, len(candidates))
        documents = [self._build_document(chunk) for chunk in candidates]
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
                "top_n": effective_top_n,
            },
        }
        if self.instruct:
            payload["parameters"]["instruct"] = self.instruct
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started_at = time.perf_counter()
        logger.info(
            "RERANK_REQUEST | provider=%s | model=%s | candidate_count=%s | top_n=%s | "
            "query_chars=%s | document_chars=%s",
            self.name,
            self.model,
            len(candidates),
            effective_top_n,
            len(query),
            sum(len(document) for document in documents),
        )

        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/reranks",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            self._log_failure(started_at, exc)
            raise RerankProviderError("qwen3 rerank request timed out") from exc
        except httpx.HTTPStatusError as exc:
            body_text = ""
            try:
                body_text = exc.response.text[:2000]
            except Exception:
                pass
            self._log_failure(
                started_at, exc, status_code=exc.response.status_code, body=body_text
            )
            raise RerankProviderError(
                f"qwen3 rerank request failed with HTTP {exc.response.status_code}: {body_text}"
            ) from exc
        except httpx.HTTPError as exc:
            self._log_failure(started_at, exc)
            raise RerankProviderError(f"qwen3 rerank request failed: {type(exc).__name__}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            self._log_failure(started_at, exc, status_code=response.status_code)
            raise RerankProviderError("qwen3 rerank response is not valid JSON") from exc

        try:
            reranked = self._parse_results(
                body,
                candidates=candidates,
                expected_count=effective_top_n,
            )
        except RerankProviderError as exc:
            self._log_failure(started_at, exc, status_code=response.status_code)
            raise

        cost_ms = int((time.perf_counter() - started_at) * 1000)
        usage = body.get("usage") if isinstance(body, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else {}
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        request_id = (
            (body.get("request_id") or body.get("id"))
            if isinstance(body, dict)
            else None
        )
        logger.info(
            "RERANK_RESPONSE | provider=%s | model=%s | returned_count=%s | cost_ms=%s | "
            "total_tokens=%s | request_id=%s",
            self.name,
            self.model,
            len(reranked),
            cost_ms,
            total_tokens,
            request_id,
        )
        return reranked

    def _build_document(self, chunk: dict[str, Any]) -> str:
        retrieval_text = str(chunk.get("retrieval_text") or "").strip()
        if retrieval_text:
            return retrieval_text[: self.chunk_max_chars]
        title = str(chunk.get("title") or "").strip()
        content = str(chunk.get("content") or "")[: self.chunk_max_chars]
        return f"标题：{title}\n正文：{content}"

    @staticmethod
    def _parse_results(
        body: Any,
        *,
        candidates: list[dict],
        expected_count: int,
    ) -> list[dict]:
        if not isinstance(body, dict):
            raise RerankProviderError("qwen3 rerank response root must be an object")
        results = body.get("results")
        if not isinstance(results, list):
            output = body.get("output")
            if isinstance(output, dict):
                results = output.get("results")
        if not isinstance(results, list):
            raise RerankProviderError("qwen3 rerank response missing results")
        if len(results) != expected_count:
            raise RerankProviderError(
                "qwen3 rerank response size mismatch: "
                f"expected={expected_count}, actual={len(results)}"
            )

        parsed: list[tuple[float, int, dict]] = []
        seen_indexes: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise RerankProviderError("qwen3 rerank result must be an object")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankProviderError("qwen3 rerank result index must be an integer")
            if index < 0 or index >= len(candidates):
                raise RerankProviderError(f"qwen3 rerank result index out of range: {index}")
            if index in seen_indexes:
                raise RerankProviderError(f"qwen3 rerank result index duplicated: {index}")
            seen_indexes.add(index)

            raw_score = item.get("relevance_score")
            if isinstance(raw_score, bool):
                raise RerankProviderError("qwen3 rerank relevance_score must be numeric")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RerankProviderError("qwen3 rerank relevance_score must be numeric") from exc
            if not math.isfinite(score):
                raise RerankProviderError("qwen3 rerank relevance_score must be finite")
            score = min(max(score, 0.0), 1.0)
            parsed.append((score, index, candidates[index]))

        parsed.sort(key=lambda item: (-item[0], item[1]))
        return [{**chunk, "rerank_score": score} for score, _, chunk in parsed]

    def _log_failure(
        self,
        started_at: float,
        error: Exception,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        cost_ms = int((time.perf_counter() - started_at) * 1000)
        logger.warning(
            "RERANK_FAILED | provider=%s | model=%s | cost_ms=%s | status_code=%s | "
            "error_type=%s | error=%s | body=%s",
            self.name,
            self.model,
            cost_ms,
            status_code,
            type(error).__name__,
            str(error),
            body,
        )
