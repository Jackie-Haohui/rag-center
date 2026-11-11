from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.chunk_relation import ChunkRelation


class ChunkRelationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_many(self, relations: list[ChunkRelation]) -> None:
        if relations:
            self.db.add_all(relations)
            self.db.flush()

    def delete_by_kb_version(self, tenant_id: str, kb_id: str, index_version: str) -> None:
        stmt = delete(ChunkRelation).where(
            ChunkRelation.tenant_id == tenant_id,
            ChunkRelation.kb_id == kb_id,
            ChunkRelation.index_version == index_version,
        )
        self.db.execute(stmt)
        self.db.flush()

    def list_targets(
        self,
        source_chunk_id: str,
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
        limit: int,
    ) -> list[str]:
        stmt = (
            select(ChunkRelation.target_chunk_id)
            .where(
                ChunkRelation.source_chunk_id == source_chunk_id,
                ChunkRelation.tenant_id == tenant_id,
                ChunkRelation.kb_id == kb_id,
                ChunkRelation.index_version == index_version,
                ChunkRelation.relation_type == "reference",
            )
            .order_by(ChunkRelation.created_at)
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def list_targets_batch(
        self,
        source_chunk_ids: list[str],
        *,
        tenant_id: str,
        allowed_kb_ids: list[str],
    ) -> dict[str, list[str]]:
        if not source_chunk_ids:
            return {}
        stmt = (
            select(ChunkRelation.source_chunk_id, ChunkRelation.target_chunk_id)
            .where(
                ChunkRelation.source_chunk_id.in_(source_chunk_ids),
                ChunkRelation.tenant_id == tenant_id,
                ChunkRelation.kb_id.in_(allowed_kb_ids),
                ChunkRelation.relation_type == "reference",
            )
            .order_by(ChunkRelation.created_at)
        )
        result: dict[str, list[str]] = {}
        for source_id, target_id in self.db.execute(stmt).all():
            targets = result.setdefault(source_id, [])
            if target_id not in targets:
                targets.append(target_id)
        return result
