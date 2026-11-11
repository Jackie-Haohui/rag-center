from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.index_versions import DEFAULT_INDEX_VERSION
from app.providers.llm.base import LLMProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.rag import (
    EvidenceOptions,
    EvidencePack,
    RetrieveEvidenceMetadata,
)
from app.schemas.research import ResearchAspect, ResearchCandidate
from app.services.evidence_pack_builder import EvidencePackBuilder, EvidenceReference

MAX_EVIDENCE_CANDIDATES = 20
MIN_BASE_POOL_CANDIDATES = 6
DEFAULT_MAX_EVIDENCE_ITEMS = 8
MAX_EVIDENCE_TEXT_CHARS = 2048
MAX_EVIDENCE_OUTPUT_CHARS = 12000
MAX_ASPECT_CHARS = 100

EVIDENCE_ORCHESTRATION_PROMPT = """你是 RAG 证据编排器，不负责回答问题。

任务：
1. 将用户问题拆成回答时必须覆盖的最少要点。
2. 只从给定 candidates 中选择能够支撑这些要点的证据。
3. core 表示直接回答要点，supporting 表示提供必要条件、范围或补充说明。
4. 不要因为主题相似就选择候选，不要选择重复证据。
5. 每个必要要点都必须输出；没有核心证据时返回空 evidence 数组。
6. 同一候选可以支撑多个要点。
7. 不生成答案，不改写证据，不输出候选中不存在的事实。

严格返回 JSON：
{"aspects":[{"aspect":"简短的问题要点","evidence":[{"index":0,"role":"core|supporting"}]}]}
"""


@dataclass(slots=True)
class EvidenceOrchestrationResult:
    pack: EvidencePack | None
    metadata: RetrieveEvidenceMetadata


class EvidenceOrchestrationService:
    def __init__(self, db: Session, llm_provider: LLMProvider | None = None) -> None:
        self.chunk_repository = ChunkRepository(db)
        self.llm_provider = llm_provider or OpenAICompatibleLLMProvider()

    async def orchestrate(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_ids: list[str],
        index_versions: dict[str, str],
        candidates: list[dict],
        retrieved_chunks: list[dict],
        options: EvidenceOptions,
        rerank_enabled: bool,
    ) -> EvidenceOrchestrationResult:
        enabled = options.enabled is True
        metadata = RetrieveEvidenceMetadata(
            enabled=enabled,
            rerank_enabled=rerank_enabled,
        )
        if not enabled:
            return EvidenceOrchestrationResult(pack=None, metadata=metadata)

        started_at = perf_counter()
        try:
            evidence_candidates, context_included = self.build_candidates(
                candidates=candidates,
                retrieved_chunks=retrieved_chunks,
            )
            metadata.executed = True
            metadata.candidate_count = len(evidence_candidates)
            metadata.context_sources_included = context_included
            if not evidence_candidates:
                pack = EvidencePack(status="missing")
                metadata.status = pack.status
                return EvidenceOrchestrationResult(pack=pack, metadata=metadata)

            max_items = options.max_items or DEFAULT_MAX_EVIDENCE_ITEMS
            payload = {
                "question": query,
                "max_evidence_items": max_items,
                "candidates": [
                    {
                        "index": index,
                        "title": candidate["title"],
                        "heading_path": candidate.get("heading_path"),
                        "text": str(
                            candidate.get("retrieval_text") or candidate.get("content") or ""
                        )[:MAX_EVIDENCE_TEXT_CHARS],
                        "retrieved_rank": candidate.get("retrieved_rank"),
                        "source": candidate["source"],
                    }
                    for index, candidate in enumerate(evidence_candidates)
                ],
            }
            raw = await self.llm_provider.chat_json(
                system_prompt=EVIDENCE_ORCHESTRATION_PROMPT,
                user_payload=payload,
                temperature=0.0,
                timeout_seconds=settings.evidence_timeout_seconds,
                log_payload=False,
            )
            aspects, selected_indexes = self._validate_model_output(
                raw,
                candidate_count=len(evidence_candidates),
                max_items=max_items,
            )
            research_aspects = [
                ResearchAspect(aspect_id=f"A{index + 1}", description=item["aspect"])
                for index, item in enumerate(aspects)
            ]
            research_candidates = [
                ResearchCandidate(
                    candidate_id=f"C{index + 1}",
                    chunk_id=str(candidate["chunk_id"]),
                    kb_id=str(candidate.get("kb_id") or ""),
                    document_id=str(candidate.get("document_id") or ""),
                    index_version=str(candidate.get("index_version") or DEFAULT_INDEX_VERSION),
                    title=str(candidate.get("title") or ""),
                    heading_path=candidate.get("heading_path"),
                    content=str(candidate.get("content") or ""),
                    retrieval_text=candidate.get("retrieval_text"),
                    source=str(candidate["source"]),
                    best_rank=candidate.get("retrieved_rank"),
                )
                for index, candidate in enumerate(evidence_candidates)
            ]
            selections = {
                research_aspects[index].aspect_id: [
                    EvidenceReference(
                        candidate_index=reference["index"],
                        role=reference["role"],
                    )
                    for reference in item["evidence"]
                ]
                for index, item in enumerate(aspects)
            }
            build_result = EvidencePackBuilder(self.chunk_repository).build(
                tenant_id=tenant_id,
                index_versions={
                    kb_id: index_versions[kb_id] for kb_id in kb_ids if kb_id in index_versions
                },
                aspects=research_aspects,
                selections=selections,
                candidates=research_candidates,
                evidence_order=selected_indexes,
                max_items=max_items,
                max_output_chars=MAX_EVIDENCE_OUTPUT_CHARS,
            )
            pack = build_result.pack
            budget_exceeded = build_result.budget_exceeded
            metadata.evidence_count = len(pack.items)
            metadata.output_chars = sum(len(item.content) for item in pack.items)
            metadata.budget_exceeded = budget_exceeded
            metadata.status = pack.status
            return EvidenceOrchestrationResult(pack=pack, metadata=metadata)
        except Exception as exc:  # evidence is an optional, isolated enhancement
            metadata.degraded = True
            metadata.error = f"{type(exc).__name__}: {exc}"
            return EvidenceOrchestrationResult(pack=None, metadata=metadata)
        finally:
            metadata.latency_ms = int((perf_counter() - started_at) * 1000)

    @classmethod
    def build_candidates(
        cls,
        *,
        candidates: list[dict],
        retrieved_chunks: list[dict],
    ) -> tuple[list[dict[str, Any]], bool]:
        top_and_context: list[dict[str, Any]] = []
        base_pool: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        context_included = False

        def add(target: list[dict[str, Any]], value: dict[str, Any]) -> bool:
            chunk_id = str(value.get("chunk_id") or "")
            content = str(value.get("content") or "").strip()
            normalized = " ".join(content.split()).casefold()
            if not chunk_id or not content or chunk_id in seen_ids or normalized in seen_content:
                return False
            seen_ids.add(chunk_id)
            seen_content.add(normalized)
            target.append(value)
            return True

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            add(
                top_and_context,
                {
                    **chunk,
                    "retrieved_rank": rank,
                    "source": str(chunk.get("retrieval_source") or "retrieval"),
                    "heading_path": cls._heading_path(chunk.get("metadata")),
                },
            )
            context = chunk.get("context") or {}
            for source in context.get("sources") or []:
                if source.get("relation") == "anchor":
                    continue
                added = add(
                    top_and_context,
                    {
                        **source,
                        "kb_id": chunk.get("kb_id"),
                        "index_version": chunk.get("index_version"),
                        "retrieval_text": source.get("content"),
                        "retrieved_rank": None,
                        "source": f"context:{source.get('relation') or 'unknown'}",
                        "heading_path": cls._heading_path(source.get("metadata")),
                    },
                )
                context_included = context_included or added

        for chunk in candidates:
            add(
                base_pool,
                {
                    **chunk,
                    "retrieved_rank": None,
                    "source": str(chunk.get("retrieval_source") or "retrieval"),
                    "heading_path": cls._heading_path(chunk.get("metadata")),
                },
            )

        reserved_base = min(MIN_BASE_POOL_CANDIDATES, len(base_pool))
        top_limit = MAX_EVIDENCE_CANDIDATES - reserved_base
        selected = top_and_context[:top_limit]
        selected.extend(base_pool[: MAX_EVIDENCE_CANDIDATES - len(selected)])
        return selected, context_included

    @staticmethod
    def _heading_path(metadata: Any) -> str | None:
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("heading_path")
        if isinstance(value, list):
            return "/".join(str(item) for item in value if str(item).strip()) or None
        if value is None:
            return None
        return str(value).strip() or None

    @staticmethod
    def _validate_model_output(
        raw: dict,
        *,
        candidate_count: int,
        max_items: int,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        if not isinstance(raw, dict) or not isinstance(raw.get("aspects"), list):
            raise ValueError("invalid evidence response: aspects must be an array")
        if not raw["aspects"]:
            raise ValueError("invalid evidence response: aspects cannot be empty")

        validated: list[dict[str, Any]] = []
        selected: list[int] = []
        selected_seen: set[int] = set()
        aspect_names: set[str] = set()
        for item in raw["aspects"]:
            if not isinstance(item, dict):
                raise ValueError("invalid evidence aspect")
            aspect = str(item.get("aspect") or "").strip()
            if not aspect or len(aspect) > MAX_ASPECT_CHARS or aspect in aspect_names:
                raise ValueError("invalid or duplicate evidence aspect")
            aspect_names.add(aspect)
            references = item.get("evidence")
            if not isinstance(references, list):
                raise ValueError("evidence references must be an array")
            local_seen: set[int] = set()
            normalized_references: list[dict[str, Any]] = []
            for reference in references:
                if not isinstance(reference, dict):
                    raise ValueError("invalid evidence reference")
                index = reference.get("index")
                role = reference.get("role")
                if type(index) is not int or index < 0 or index >= candidate_count:
                    raise ValueError("evidence index out of range")
                if index in local_seen:
                    raise ValueError("duplicate evidence index in aspect")
                if role not in {"core", "supporting"}:
                    raise ValueError("invalid evidence role")
                local_seen.add(index)
                normalized_references.append({"index": index, "role": role})
                if index not in selected_seen:
                    selected_seen.add(index)
                    selected.append(index)
            validated.append({"aspect": aspect, "evidence": normalized_references})
        if len(selected) > max_items:
            raise ValueError("selected evidence exceeds max_items")
        return validated, selected


def prompt_sha256() -> str:
    import hashlib

    return hashlib.sha256(EVIDENCE_ORCHESTRATION_PROMPT.encode("utf-8")).hexdigest()


def model_output_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
