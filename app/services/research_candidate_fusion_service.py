from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.index_versions import DEFAULT_INDEX_VERSION
from app.core.logging import get_logger
from app.schemas.research import ResearchAspect, ResearchCandidate
from app.services.research_limits import MAX_RESEARCH_CANDIDATES
from app.services.retrieve_once_service import RetrieveOnceResult
from app.utils.markdown_splitter import normalize_content

logger = get_logger(__name__)
RRF_K = 60


@dataclass(slots=True)
class CandidateFusionResult:
    candidates: list[ResearchCandidate]
    before_count: int
    deduplicated_count: int
    aspect_candidate_ids: dict[str, list[str]]


class ResearchCandidateFusionService:
    def fuse(
        self,
        *,
        results: list[RetrieveOnceResult],
        aspects: list[ResearchAspect],
        max_candidates: int = MAX_RESEARCH_CANDIDATES,
    ) -> CandidateFusionResult:
        merged: list[ResearchCandidate] = []
        by_primary_key: dict[tuple[str, str, str], int] = {}
        by_content_key: dict[tuple[str, str, str], int] = {}
        contribution_keys: set[tuple[int, str]] = set()
        before_count = 0

        for result in results:
            for rank, chunk in enumerate(result.retrieved_chunks, start=1):
                before_count += 1
                self._merge_occurrence(
                    merged=merged,
                    by_primary_key=by_primary_key,
                    by_content_key=by_content_key,
                    contribution_keys=contribution_keys,
                    chunk=chunk,
                    result=result,
                    rank=rank,
                    source=str(chunk.get("retrieval_source") or "retrieval"),
                )
                context = chunk.get("context") or {}
                for source in context.get("sources") or []:
                    relation = str(source.get("relation") or "")
                    if relation == "anchor" or not source.get("content"):
                        continue
                    before_count += 1
                    context_chunk = {
                        "kb_id": chunk.get("kb_id"),
                        "document_id": source.get("document_id") or chunk.get("document_id"),
                        "chunk_id": source.get("chunk_id"),
                        "title": source.get("title") or chunk.get("title"),
                        "content": source.get("content"),
                        "retrieval_text": None,
                        "index_version": chunk.get("index_version") or DEFAULT_INDEX_VERSION,
                        "metadata": source.get("metadata") or {},
                    }
                    self._merge_occurrence(
                        merged=merged,
                        by_primary_key=by_primary_key,
                        by_content_key=by_content_key,
                        contribution_keys=contribution_keys,
                        chunk=context_chunk,
                        result=result,
                        rank=rank,
                        source=f"context:{relation}",
                        context_relation=relation,
                        anchor_chunk_id=str(chunk.get("chunk_id") or ""),
                    )

        selected = self._fair_select(merged, aspects, max_candidates=max_candidates)
        for index, candidate in enumerate(selected, start=1):
            candidate.candidate_id = f"C{index}"
        aspect_candidate_ids = {
            aspect.aspect_id: [
                candidate.candidate_id
                for candidate in selected
                if aspect.aspect_id in candidate.aspect_ids
            ]
            for aspect in aspects
        }
        logger.info(
            "RESEARCH_CANDIDATE_FUSION | before_count=%s | candidate_count=%s | "
            "deduplicated_count=%s",
            before_count,
            len(selected),
            max(0, before_count - len(merged)),
        )
        return CandidateFusionResult(
            candidates=selected,
            before_count=before_count,
            deduplicated_count=max(0, before_count - len(merged)),
            aspect_candidate_ids=aspect_candidate_ids,
        )

    @staticmethod
    def _merge_occurrence(
        *,
        merged: list[ResearchCandidate],
        by_primary_key: dict[tuple[str, str, str], int],
        by_content_key: dict[tuple[str, str, str], int],
        contribution_keys: set[tuple[int, str]],
        chunk: dict[str, Any],
        result: RetrieveOnceResult,
        rank: int,
        source: str,
        context_relation: str | None = None,
        anchor_chunk_id: str | None = None,
    ) -> None:
        kb_id = str(chunk.get("kb_id") or "")
        index_version = str(chunk.get("index_version") or DEFAULT_INDEX_VERSION)
        chunk_id = str(chunk.get("chunk_id") or "")
        content = str(chunk.get("content") or "")
        primary_key = (kb_id, index_version, chunk_id)
        content_key = (kb_id, index_version, normalize_content(content))
        existing_index = by_primary_key.get(primary_key) if chunk_id else None
        if existing_index is None and content_key[2]:
            existing_index = by_content_key.get(content_key)

        if existing_index is None:
            metadata = dict(chunk.get("metadata") or {})
            metadata["sources"] = [source]
            if anchor_chunk_id:
                metadata["anchor_chunk_ids"] = [anchor_chunk_id]
            heading_path = metadata.get("heading_path")
            if isinstance(heading_path, list):
                heading_path = " > ".join(str(item) for item in heading_path)
            candidate = ResearchCandidate(
                chunk_id=chunk_id,
                kb_id=kb_id,
                document_id=str(chunk.get("document_id") or ""),
                index_version=index_version,
                title=str(chunk.get("title") or ""),
                heading_path=str(heading_path) if heading_path else None,
                content=content,
                retrieval_text=chunk.get("retrieval_text"),
                source=source,
                aspect_ids=list(dict.fromkeys(result.aspect_ids)),
                query_ids=[result.query_id],
                rounds=[result.round],
                best_rank=rank,
                context_relation=context_relation,
                metadata=metadata,
            )
            merged.append(candidate)
            existing_index = len(merged) - 1
            if chunk_id:
                by_primary_key[primary_key] = existing_index
            if content_key[2]:
                by_content_key[content_key] = existing_index
        else:
            candidate = merged[existing_index]
            candidate.aspect_ids = list(dict.fromkeys([*candidate.aspect_ids, *result.aspect_ids]))
            candidate.query_ids = list(dict.fromkeys([*candidate.query_ids, result.query_id]))
            candidate.rounds = sorted({*candidate.rounds, result.round})
            candidate.best_rank = min(candidate.best_rank or rank, rank)
            sources = candidate.metadata.setdefault("sources", [])
            if source not in sources:
                sources.append(source)
            if anchor_chunk_id:
                anchors = candidate.metadata.setdefault("anchor_chunk_ids", [])
                if anchor_chunk_id not in anchors:
                    anchors.append(anchor_chunk_id)

        contribution_key = (existing_index, result.query_id)
        if contribution_key not in contribution_keys:
            merged[existing_index].fusion_score += 1 / (RRF_K + rank)
            contribution_keys.add(contribution_key)

    @staticmethod
    def _fair_select(
        candidates: list[ResearchCandidate],
        aspects: list[ResearchAspect],
        *,
        max_candidates: int,
    ) -> list[ResearchCandidate]:
        if not candidates or max_candidates <= 0:
            return []
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (-item[1].fusion_score, item[1].best_rank or 10**9, item[0]),
        )
        per_aspect = {
            aspect.aspect_id: [item for _, item in ranked if aspect.aspect_id in item.aspect_ids]
            for aspect in aspects
        }
        selected: list[ResearchCandidate] = []
        selected_keys: set[tuple[str, str, str]] = set()

        def add(candidate: ResearchCandidate) -> None:
            key = (
                candidate.kb_id,
                candidate.index_version,
                candidate.chunk_id or candidate.content,
            )
            if key not in selected_keys and len(selected) < max_candidates:
                selected.append(candidate)
                selected_keys.add(key)

        for target_count in (1, 2):
            for aspect in aspects:
                options = per_aspect[aspect.aspect_id]
                if len(options) >= target_count:
                    add(options[target_count - 1])
        for _, candidate in ranked:
            if len(candidate.aspect_ids) > 1:
                add(candidate)
        for _, candidate in ranked:
            add(candidate)
        return selected
