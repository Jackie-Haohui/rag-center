from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.chunk_relation import ChunkRelation
from app.repositories.chunk_relation_repository import ChunkRelationRepository
from app.repositories.chunk_repository import ChunkRepository
from app.utils.id_generator import generate_uuid
from app.utils.markdown_splitter import normalize_content

DOCUMENT_REFERENCE_RE = re.compile(
    r"《(?P<document>[^》]+)》"
    r"(?:第\s*(?P<section>[\d.]+)\s*节)?"
    r"(?:[“\"](?P<label>[^”\"]+)[”\"])?"
)
LOCAL_REFERENCE_RE = re.compile(
    r"本文件第\s*(?P<section>[\d.]+)\s*节(?:[“\"](?P<label>[^”\"]+)[”\"])?"
)
MARKDOWN_REFERENCE_RE = re.compile(
    r"\[(?P<label>[^\]]+)]\((?P<document>[^)#]+)(?:#(?P<section>[^)]+))?\)"
)


@dataclass(frozen=True, slots=True)
class ParsedReference:
    raw: str
    document_title: str | None
    section: str | None
    label: str | None


def normalize_document_title(value: str) -> str:
    stem = Path(value.strip()).stem
    return normalize_content(stem)


def parse_explicit_references(content: str) -> list[ParsedReference]:
    references: list[ParsedReference] = []
    occupied: list[tuple[int, int]] = []
    for match in DOCUMENT_REFERENCE_RE.finditer(content):
        references.append(
            ParsedReference(
                raw=match.group(0),
                document_title=match.group("document"),
                section=match.group("section"),
                label=match.group("label"),
            )
        )
        occupied.append(match.span())
    for match in LOCAL_REFERENCE_RE.finditer(content):
        references.append(
            ParsedReference(
                raw=match.group(0),
                document_title=None,
                section=match.group("section"),
                label=match.group("label"),
            )
        )
        occupied.append(match.span())
    for match in MARKDOWN_REFERENCE_RE.finditer(content):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        references.append(
            ParsedReference(
                raw=match.group(0),
                document_title=match.group("document"),
                section=match.group("section"),
                label=match.group("label"),
            )
        )
    return references


class ReferenceRelationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.chunk_repository = ChunkRepository(db)
        self.relation_repository = ChunkRelationRepository(db)

    def rebuild(self, *, tenant_id: str, kb_id: str, index_version: str) -> int:
        chunks = self.chunk_repository.list_by_kb_version(tenant_id, kb_id, index_version)
        self.relation_repository.delete_by_kb_version(tenant_id, kb_id, index_version)
        documents: dict[str, list[str]] = defaultdict(list)
        chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            normalized_title = normalize_document_title(chunk.title)
            if chunk.document_id not in documents[normalized_title]:
                documents[normalized_title].append(chunk.document_id)
            chunks_by_document[chunk.document_id].append(chunk)

        relations: list[ChunkRelation] = []
        seen: set[tuple[str, str]] = set()
        for source in chunks:
            parsed = parse_explicit_references(source.content)
            if not parsed:
                continue
            source_metadata = dict(source.metadata_ or {})
            source_metadata["reference_texts"] = [item.raw for item in parsed]
            source.metadata_ = source_metadata

            for reference in parsed:
                target_document_ids = self._resolve_document_ids(
                    reference,
                    source_document_id=source.document_id,
                    documents=documents,
                )
                if len(target_document_ids) != 1:
                    continue
                target = self._resolve_target_chunk(
                    chunks_by_document[target_document_ids[0]],
                    reference,
                )
                if target is None or target.id == source.id:
                    continue
                key = (source.id, target.id)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    ChunkRelation(
                        id=generate_uuid(),
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        index_version=index_version,
                        source_chunk_id=source.id,
                        target_chunk_id=target.id,
                        relation_type="reference",
                        metadata_={"reference_text": reference.raw},
                    )
                )

        self.relation_repository.add_many(relations)
        return len(relations)

    @staticmethod
    def _resolve_document_ids(
        reference: ParsedReference,
        *,
        source_document_id: str,
        documents: dict[str, list[str]],
    ) -> list[str]:
        if reference.document_title is None:
            return [source_document_id]
        return documents.get(normalize_document_title(reference.document_title), [])

    @staticmethod
    def _resolve_target_chunk(
        chunks: list[Chunk], reference: ParsedReference
    ) -> Chunk | None:
        candidates = chunks
        normalized_label = normalize_content(reference.label or "")
        normalized_section = normalize_content(reference.section or "")
        if normalized_label:
            candidates = [
                chunk
                for chunk in candidates
                if normalized_label
                in normalize_content(str((chunk.metadata_ or {}).get("heading_path") or ""))
            ]
        elif normalized_section:
            candidates = [
                chunk
                for chunk in candidates
                if normalized_section
                in normalize_content(str((chunk.metadata_ or {}).get("heading_path") or ""))
            ]
        if not candidates:
            return None

        section_ids = {chunk.section_id for chunk in candidates if chunk.section_id}
        if len(section_ids) > 1:
            return None
        return min(
            candidates,
            key=lambda chunk: chunk.order_index if chunk.order_index is not None else 10**9,
        )
