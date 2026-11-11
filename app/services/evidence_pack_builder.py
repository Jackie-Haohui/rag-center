from __future__ import annotations

from dataclasses import dataclass

from app.repositories.chunk_repository import ChunkRepository
from app.schemas.rag import (
    EvidenceGroup,
    EvidenceGroupItem,
    EvidenceItem,
    EvidencePack,
    EvidenceRole,
    EvidenceStatus,
)
from app.schemas.research import ResearchAspect, ResearchCandidate

DEFAULT_MAX_OUTPUT_CHARS = 12000


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    candidate_index: int
    role: EvidenceRole


@dataclass(slots=True)
class EvidencePackBuildResult:
    pack: EvidencePack
    evidence_id_by_candidate_index: dict[int, str]
    budget_exceeded: bool = False


def recompute_evidence_status(
    *,
    aspects: list[ResearchAspect],
    groups: list[EvidenceGroup],
) -> tuple[EvidenceStatus, list[str]]:
    covered_by_name = {group.aspect: group.covered for group in groups}
    missing_aspects = [
        aspect.description
        for aspect in aspects
        if not covered_by_name.get(aspect.description, False)
    ]
    covered_count = len(aspects) - len(missing_aspects)
    if aspects and covered_count == len(aspects):
        return "complete", []
    if covered_count == 0:
        return "missing", missing_aspects
    return "partial", missing_aspects


class EvidencePackBuilder:
    """Builds evidence only after repository-backed tenant and version validation."""

    def __init__(self, chunk_repository: ChunkRepository) -> None:
        self.chunk_repository = chunk_repository

    def build(
        self,
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        aspects: list[ResearchAspect],
        selections: dict[str, list[EvidenceReference]],
        candidates: list[ResearchCandidate],
        evidence_order: list[int] | None = None,
        max_items: int | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> EvidencePackBuildResult:
        selected_indexes = self._selected_indexes(
            aspects=aspects,
            selections=selections,
            evidence_order=evidence_order,
        )
        if max_items is not None:
            selected_indexes = selected_indexes[:max_items]
        selected_ids = [candidates[index].chunk_id for index in selected_indexes]
        if any(not chunk_id for chunk_id in selected_ids):
            raise PermissionError("selected evidence has no repository chunk id")
        unique_ids = list(dict.fromkeys(selected_ids))
        chunks = self.chunk_repository.get_by_ids_in_versioned_scope(
            unique_ids,
            tenant_id=tenant_id,
            index_versions=index_versions,
        )
        by_id = {chunk.id: chunk for chunk in chunks}
        if len(by_id) != len(unique_ids) or any(chunk_id not in by_id for chunk_id in unique_ids):
            raise PermissionError("selected evidence failed tenant, kb, or index validation")

        evidence_id_by_index: dict[int, str] = {}
        items: list[EvidenceItem] = []
        output_chars = 0
        budget_exceeded = False
        for index in selected_indexes:
            candidate = candidates[index]
            chunk = by_id[candidate.chunk_id]
            content = chunk.content
            if output_chars + len(content) > max_output_chars:
                budget_exceeded = True
                continue
            evidence_id = f"E{len(items) + 1}"
            evidence_id_by_index[index] = evidence_id
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=chunk.id,
                    kb_id=chunk.kb_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    heading_path=self._heading_path(chunk.metadata_ or {}),
                    index_version=chunk.index_version,
                    content=content,
                    source=candidate.source,
                    retrieved_rank=candidate.best_rank,
                    rounds=candidate.rounds,
                    query_ids=candidate.query_ids,
                )
            )
            output_chars += len(content)

        groups: list[EvidenceGroup] = []
        for aspect in aspects:
            references = [
                EvidenceGroupItem(
                    evidence_id=evidence_id_by_index[reference.candidate_index],
                    role=reference.role,
                )
                for reference in selections.get(aspect.aspect_id, [])
                if reference.candidate_index in evidence_id_by_index
            ]
            groups.append(
                EvidenceGroup(
                    aspect=aspect.description,
                    evidence=references,
                    covered=any(reference.role == "core" for reference in references),
                )
            )

        status, missing_aspects = recompute_evidence_status(aspects=aspects, groups=groups)
        return EvidencePackBuildResult(
            pack=EvidencePack(
                status=status,
                missing_aspects=missing_aspects,
                groups=groups,
                items=items,
            ),
            evidence_id_by_candidate_index=evidence_id_by_index,
            budget_exceeded=budget_exceeded,
        )

    @staticmethod
    def _selected_indexes(
        *,
        aspects: list[ResearchAspect],
        selections: dict[str, list[EvidenceReference]],
        evidence_order: list[int] | None,
    ) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for index in evidence_order or []:
            if index not in seen:
                ordered.append(index)
                seen.add(index)
        for aspect in aspects:
            for reference in selections.get(aspect.aspect_id, []):
                if reference.candidate_index not in seen:
                    ordered.append(reference.candidate_index)
                    seen.add(reference.candidate_index)
        return ordered

    @staticmethod
    def _heading_path(metadata: dict) -> str | None:
        value = metadata.get("heading_path")
        if isinstance(value, list):
            return "/".join(str(item) for item in value if str(item).strip()) or None
        if value is None:
            return None
        return str(value).strip() or None
