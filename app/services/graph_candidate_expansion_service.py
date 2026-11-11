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
from app.schemas.rag import RetrieveGraphInjectionMetadata
from app.utils.markdown_splitter import is_low_information_content

logger = get_logger(__name__)

MAX_GRAPH_SEEDS = 10
MAX_INJECTED_CHUNKS_PER_SEED = 3
MAX_TOTAL_CANDIDATES = 30


@dataclass(slots=True)
class GraphCandidateExpansionResult:
    candidates: list[dict[str, Any]]
    metadata: RetrieveGraphInjectionMetadata


class GraphCandidateExpansionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.chunk_repository = ChunkRepository(db)
        self.relation_repository = ChunkRelationRepository(db)

    async def expand_candidates(
        self,
        *,
        tenant_id: str,
        kb_ids: list[str],
        seeds: list[dict[str, Any]],
    ) -> GraphCandidateExpansionResult:
        started_at = perf_counter()
        if not seeds or all(seed.get("index_version") == LEGACY_INDEX_VERSION for seed in seeds):
            return GraphCandidateExpansionResult(
                candidates=seeds,
                metadata=RetrieveGraphInjectionMetadata(
                    executed=False,
                    seed_count=len(seeds),
                    candidate_count=len(seeds),
                    injected_count=0,
                    latency_ms=int((perf_counter() - started_at) * 1000),
                ),
            )

        contextual_seeds = [
            seed
            for seed in seeds
            if seed.get("index_version") != LEGACY_INDEX_VERSION
            and str((seed.get("metadata") or {}).get("chunk_type") or "prose")
            != "reference_pointer"
        ]
        if not contextual_seeds:
            return GraphCandidateExpansionResult(
                candidates=seeds,
                metadata=RetrieveGraphInjectionMetadata(
                    executed=False,
                    seed_count=len(seeds),
                    candidate_count=len(seeds),
                    injected_count=0,
                    latency_ms=int((perf_counter() - started_at) * 1000),
                ),
            )

        try:
            graph_seeds = contextual_seeds[:MAX_GRAPH_SEEDS]
            seen_ids = {str(seed.get("chunk_id") or "") for seed in seeds if seed.get("chunk_id")}
            merged = list(seeds)
            injected_count = 0

            target_ids_by_source = self.relation_repository.list_targets_batch(
                [str(seed.get("chunk_id") or "") for seed in graph_seeds],
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

            neighbor_anchors = [
                seed
                for seed in graph_seeds
                if str((seed.get("metadata") or {}).get("chunk_type") or "prose") == "prose"
            ]
            neighbor_map = {
                (
                    chunk.kb_id,
                    chunk.document_id,
                    chunk.index_version,
                    chunk.section_id,
                    chunk.order_index,
                ): chunk
                for chunk in self.chunk_repository.find_neighbors_batch(
                    neighbor_anchors,
                    tenant_id=tenant_id,
                    allowed_kb_ids=kb_ids,
                )
            }
            parent_anchors = [seed for seed in graph_seeds if seed.get("parent_section_id")]
            parent_map = {
                (
                    chunk.kb_id,
                    chunk.document_id,
                    chunk.index_version,
                    chunk.section_id,
                ): chunk
                for chunk in self.chunk_repository.first_in_sections_batch(
                    parent_anchors,
                    tenant_id=tenant_id,
                    allowed_kb_ids=kb_ids,
                )
            }

            for seed in graph_seeds:
                if len(merged) >= MAX_TOTAL_CANDIDATES:
                    break
                seed_injected = 0
                for chunk, relation in self._collect_related_chunks(
                    seed=seed,
                    target_ids_by_source=target_ids_by_source,
                    target_map=target_map,
                    neighbor_map=neighbor_map,
                    parent_map=parent_map,
                    allowed_kb_ids=set(kb_ids),
                ):
                    if seed_injected >= MAX_INJECTED_CHUNKS_PER_SEED:
                        break
                    if len(merged) >= MAX_TOTAL_CANDIDATES:
                        break
                    if chunk.id in seen_ids:
                        continue
                    if is_low_information_content(chunk.content):
                        continue
                    seen_ids.add(chunk.id)
                    merged.append(self._chunk_to_candidate(chunk, relation=relation))
                    injected_count += 1
                    seed_injected += 1

            return GraphCandidateExpansionResult(
                candidates=merged,
                metadata=RetrieveGraphInjectionMetadata(
                    executed=True,
                    seed_count=len(graph_seeds),
                    candidate_count=len(merged),
                    injected_count=injected_count,
                    latency_ms=int((perf_counter() - started_at) * 1000),
                ),
            )
        except Exception as exc:
            logger.exception(
                "GRAPH_INJECTION_DEGRADED | tenant_id=%s | kb_ids=%s",
                tenant_id,
                kb_ids,
            )
            return GraphCandidateExpansionResult(
                candidates=seeds,
                metadata=RetrieveGraphInjectionMetadata(
                    executed=True,
                    seed_count=min(len(contextual_seeds), MAX_GRAPH_SEEDS),
                    candidate_count=len(seeds),
                    injected_count=0,
                    latency_ms=int((perf_counter() - started_at) * 1000),
                    degraded=True,
                    error=str(exc)[:500],
                ),
            )

    def _collect_related_chunks(
        self,
        *,
        seed: dict[str, Any],
        target_ids_by_source: dict[str, list[str]],
        target_map: dict[str, Chunk],
        neighbor_map: dict[tuple, Chunk],
        parent_map: dict[tuple, Chunk],
        allowed_kb_ids: set[str],
    ) -> list[tuple[Chunk, str]]:
        kb_id = str(seed.get("kb_id") or "")
        index_version = str(seed.get("index_version") or "")
        document_id = str(seed.get("document_id") or "")
        section_id = seed.get("section_id")
        order_index = seed.get("order_index")
        chunk_id = str(seed.get("chunk_id") or "")
        related: list[tuple[Chunk, str]] = []

        for target_id in target_ids_by_source.get(chunk_id, []):
            target = target_map.get(target_id)
            if target is None:
                continue
            if target.kb_id not in allowed_kb_ids or target.index_version != index_version:
                logger.warning(
                    "GRAPH_INJECTION_REJECTED | reason=cross_scope_reference | "
                    "tenant_seed=%s | target_chunk=%s",
                    chunk_id,
                    target_id,
                )
                continue
            related.append((target, "reference"))

        parent_section_id = seed.get("parent_section_id")
        if parent_section_id:
            parent = parent_map.get((kb_id, document_id, index_version, str(parent_section_id)))
            if parent is not None:
                related.append((parent, "parent_section"))

        chunk_type = str((seed.get("metadata") or {}).get("chunk_type") or "prose")
        if isinstance(order_index, int) and section_id and chunk_type == "prose":
            previous_key = (kb_id, document_id, index_version, section_id, order_index - 1)
            next_key = (kb_id, document_id, index_version, section_id, order_index + 1)
            if previous_key in neighbor_map:
                related.append((neighbor_map[previous_key], "previous"))
            if next_key in neighbor_map:
                related.append((neighbor_map[next_key], "next"))

        return related

    @staticmethod
    def _chunk_to_candidate(chunk: Chunk, *, relation: str) -> dict[str, Any]:
        metadata = dict(chunk.metadata_ or {})
        metadata["injection_source"] = relation
        return {
            "kb_id": chunk.kb_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "title": chunk.title,
            "content": chunk.content,
            "retrieval_text": chunk.retrieval_text,
            "index_version": chunk.index_version,
            "section_id": chunk.section_id,
            "parent_section_id": chunk.parent_section_id,
            "order_index": chunk.order_index,
            "score": 0.0,
            "metadata": metadata,
            "retrieval_source": "graph_injection",
        }
