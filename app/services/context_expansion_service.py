from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.core.index_versions import LEGACY_INDEX_VERSION
from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.repositories.chunk_relation_repository import ChunkRelationRepository
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.rag import RetrieveContextMetadata
from app.utils.markdown_splitter import is_low_information_content, normalize_content

logger = get_logger(__name__)

MAX_NEIGHBORS_PER_ANCHOR = 2
MAX_REFERENCE_TARGETS_PER_ANCHOR = 3
MAX_SUPPLEMENTAL_CHUNKS_PER_ANCHOR = 3
MAX_CONTEXT_CHARS_PER_ANCHOR = 4096


@dataclass(slots=True)
class ContextExpansionResult:
    anchors: list[dict[str, Any]]
    metadata: RetrieveContextMetadata


class ContextExpansionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.chunk_repository = ChunkRepository(db)
        self.relation_repository = ChunkRelationRepository(db)

    async def expand(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        anchors: list[dict[str, Any]],
    ) -> ContextExpansionResult:
        started_at = perf_counter()
        expanded: list[dict[str, Any]] = []
        supplemental_count = 0
        expanded_anchor_count = 0
        deduplicated_count = 0
        executed = any(anchor.get("index_version") != LEGACY_INDEX_VERSION for anchor in anchors)
        try:
            allowed_kb_ids = set(kb_ids)
            contextual_anchors = [
                anchor
                for anchor in anchors
                if anchor.get("index_version") != LEGACY_INDEX_VERSION
                and str(anchor.get("kb_id") or "") in allowed_kb_ids
            ]
            if not contextual_anchors:
                return ContextExpansionResult(
                    anchors=[{**anchor, "context": None} for anchor in anchors],
                    metadata=RetrieveContextMetadata(
                        executed=False,
                        anchor_count=len(anchors),
                        latency_ms=int((perf_counter() - started_at) * 1000),
                    ),
                )
            neighbor_anchors = [
                anchor
                for anchor in contextual_anchors
                if str((anchor.get("metadata") or {}).get("chunk_type") or "prose") == "prose"
            ]
            neighbors = self.chunk_repository.find_neighbors_batch(
                neighbor_anchors,
                tenant_id=tenant_id,
                allowed_kb_ids=kb_ids,
            )
            neighbor_map = {
                (
                    chunk.kb_id,
                    chunk.document_id,
                    chunk.index_version,
                    chunk.section_id,
                    chunk.order_index,
                ): chunk
                for chunk in neighbors
            }
            parent_anchors = [
                anchor for anchor in contextual_anchors if self._needs_parent_context(anchor)
            ]
            parents = self.chunk_repository.first_in_sections_batch(
                parent_anchors,
                tenant_id=tenant_id,
                allowed_kb_ids=kb_ids,
            )
            parent_map = {
                (
                    chunk.kb_id,
                    chunk.document_id,
                    chunk.index_version,
                    chunk.section_id,
                ): chunk
                for chunk in parents
            }
            target_ids_by_source = self.relation_repository.list_targets_batch(
                [str(anchor.get("chunk_id") or "") for anchor in contextual_anchors],
                tenant_id=tenant_id,
                allowed_kb_ids=kb_ids,
            )
            target_ids = [
                target_id for values in target_ids_by_source.values() for target_id in values
            ]
            target_map = {
                chunk.id: chunk
                for chunk in self.chunk_repository.get_by_ids_in_allowed_scope(
                    target_ids,
                    tenant_id=tenant_id,
                    allowed_kb_ids=kb_ids,
                )
            }
            for anchor in anchors:
                if anchor.get("index_version") == LEGACY_INDEX_VERSION:
                    expanded.append({**anchor, "context": None})
                    continue
                item, added, deduplicated = self._expand_anchor(
                    tenant_id=tenant_id,
                    allowed_kb_ids=set(kb_ids),
                    anchor=anchor,
                    neighbor_map=neighbor_map,
                    parent_map=parent_map,
                    target_ids_by_source=target_ids_by_source,
                    target_map=target_map,
                )
                expanded.append(item)
                supplemental_count += added
                deduplicated_count += deduplicated
                if added:
                    expanded_anchor_count += 1
        except Exception as exc:
            logger.exception(
                "CONTEXT_EXPANSION_DEGRADED | tenant_id=%s | kb_ids=%s",
                tenant_id,
                kb_ids,
            )
            return ContextExpansionResult(
                anchors=[{**anchor, "context": None} for anchor in anchors],
                metadata=RetrieveContextMetadata(
                    executed=executed,
                    anchor_count=len(anchors),
                    latency_ms=int((perf_counter() - started_at) * 1000),
                    degraded=True,
                    error=str(exc)[:500],
                ),
            )

        return ContextExpansionResult(
            anchors=expanded,
            metadata=RetrieveContextMetadata(
                executed=executed,
                anchor_count=len(anchors),
                expanded_anchor_count=expanded_anchor_count,
                supplemental_chunk_count=supplemental_count,
                deduplicated_count=deduplicated_count,
                latency_ms=int((perf_counter() - started_at) * 1000),
            ),
        )

    def _expand_anchor(
        self,
        *,
        tenant_id: str,
        allowed_kb_ids: set[str],
        anchor: dict[str, Any],
        neighbor_map: dict[tuple, Chunk],
        parent_map: dict[tuple, Chunk],
        target_ids_by_source: dict[str, list[str]],
        target_map: dict[str, Chunk],
    ) -> tuple[dict[str, Any], int, int]:
        kb_id = str(anchor.get("kb_id") or "")
        if kb_id not in allowed_kb_ids:
            raise PermissionError(f"anchor knowledge base is not authorized: {kb_id}")
        index_version = str(anchor.get("index_version") or "")
        document_id = str(anchor.get("document_id") or "")
        section_id = anchor.get("section_id")
        order_index = anchor.get("order_index")
        anchor_content = str(anchor.get("content") or "")
        sources: list[dict[str, Any]] = [
            {
                "chunk_id": str(anchor.get("chunk_id") or ""),
                "relation": "anchor",
                "content": anchor_content,
                "document_id": document_id,
                "title": anchor.get("title"),
                "metadata": anchor.get("metadata") or {},
            }
        ]
        seen_ids = {str(anchor.get("chunk_id") or "")}
        seen_content = {normalize_content(anchor_content)}
        current_chars = len(anchor_content)
        deduplicated = 0

        candidates: list[tuple[str, Chunk]] = []
        chunk_type = str((anchor.get("metadata") or {}).get("chunk_type") or "prose")

        for target_id in target_ids_by_source.get(str(anchor.get("chunk_id") or ""), [])[
            :MAX_REFERENCE_TARGETS_PER_ANCHOR
        ]:
            target = target_map.get(target_id)
            if target is None:
                continue
            if target.kb_id != kb_id or target.index_version != index_version:
                raise PermissionError("reference target crossed knowledge base or index version")
            candidates.append(("reference", target))

        parent_section_id = anchor.get("parent_section_id")
        if parent_section_id and self._needs_parent_context(anchor):
            parent = parent_map.get((kb_id, document_id, index_version, str(parent_section_id)))
            if parent is not None:
                candidates.append(("parent_section", parent))

        if isinstance(order_index, int) and section_id and chunk_type == "prose":
            previous_key = (kb_id, document_id, index_version, section_id, order_index - 1)
            next_key = (kb_id, document_id, index_version, section_id, order_index + 1)
            neighbor_candidates: list[tuple[str, Chunk]] = []
            if previous_key in neighbor_map:
                neighbor_candidates.append(("previous", neighbor_map[previous_key]))
            if next_key in neighbor_map:
                neighbor_candidates.append(("next", neighbor_map[next_key]))
            candidates.extend(neighbor_candidates[:MAX_NEIGHBORS_PER_ANCHOR])

        supplemental = 0
        for relation, chunk in candidates:
            if supplemental >= MAX_SUPPLEMENTAL_CHUNKS_PER_ANCHOR:
                break
            content = chunk.content.strip()
            if is_low_information_content(content):
                continue
            normalized = normalize_content(content)
            if chunk.id in seen_ids or normalized in seen_content:
                deduplicated += 1
                continue
            extra_chars = len(content) + 2
            if current_chars + extra_chars > MAX_CONTEXT_CHARS_PER_ANCHOR:
                break
            seen_ids.add(chunk.id)
            seen_content.add(normalized)
            current_chars += extra_chars
            supplemental += 1
            sources.append(
                {
                    "chunk_id": chunk.id,
                    "relation": relation,
                    "content": content,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "metadata": chunk.metadata_ or {},
                }
            )

        context = {
            "content": "\n\n".join(str(source["content"]) for source in sources),
            "chunk_ids": [str(source["chunk_id"]) for source in sources],
            "sources": sources,
            "degraded": False,
            "error": None,
        }
        return {**anchor, "context": context}, supplemental, deduplicated

    @staticmethod
    def _needs_parent_context(anchor: dict[str, Any]) -> bool:
        metadata = anchor.get("metadata") or {}
        heading_level = metadata.get("heading_level")
        if not isinstance(heading_level, int) or heading_level < 3:
            return False
        content = str(anchor.get("content") or "")
        markers = ("本节", "上述", "前述", "以下", "该规则", "其中", "适用范围")
        return any(marker in content for marker in markers)
