from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_many(self, chunks: list[Chunk]) -> None:
        self.db.add_all(chunks)
        self.db.flush()

    def count_by_document_id(self, document_id: str, index_version: str | None = None) -> int:
        stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        if index_version is not None:
            stmt = stmt.where(Chunk.index_version == index_version)
        return int(self.db.scalar(stmt) or 0)

    def count_by_kb_id(
        self, tenant_id: str, kb_id: str, index_version: str | None = None
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.tenant_id == tenant_id, Chunk.kb_id == kb_id)
        )
        if index_version is not None:
            stmt = stmt.where(Chunk.index_version == index_version)
        return int(self.db.scalar(stmt) or 0)

    def delete_by_document_id(self, document_id: str, index_version: str | None = None) -> None:
        stmt = delete(Chunk).where(Chunk.document_id == document_id)
        if index_version is not None:
            stmt = stmt.where(Chunk.index_version == index_version)
        self.db.execute(stmt)
        self.db.flush()

    def delete_by_kb_version(self, tenant_id: str, kb_id: str, index_version: str) -> None:
        stmt = delete(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.kb_id == kb_id,
            Chunk.index_version == index_version,
        )
        self.db.execute(stmt)
        self.db.flush()

    def list_by_kb_version(
        self, tenant_id: str, kb_id: str, index_version: str
    ) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.index_version == index_version,
            )
            .order_by(Chunk.document_id, Chunk.order_index)
        )
        return list(self.db.scalars(stmt).all())

    def get_by_ids_in_scope(
        self,
        chunk_ids: list[str],
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str,
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(
            Chunk.id.in_(chunk_ids),
            Chunk.tenant_id == tenant_id,
            Chunk.kb_id == kb_id,
            Chunk.index_version == index_version,
        )
        return list(self.db.scalars(stmt).all())

    def find_neighbors(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        document_id: str,
        index_version: str,
        order_indexes: list[int],
        section_id: str | None = None,
    ) -> list[Chunk]:
        if not order_indexes:
            return []
        stmt = select(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.kb_id == kb_id,
            Chunk.document_id == document_id,
            Chunk.index_version == index_version,
            Chunk.order_index.in_(order_indexes),
        )
        if section_id is not None:
            stmt = stmt.where(Chunk.section_id == section_id)
        return list(self.db.scalars(stmt).all())

    def first_in_section(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        document_id: str,
        index_version: str,
        section_id: str,
    ) -> Chunk | None:
        stmt = (
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id == kb_id,
                Chunk.document_id == document_id,
                Chunk.index_version == index_version,
                Chunk.section_id == section_id,
            )
            .order_by(Chunk.order_index)
            .limit(1)
        )
        return self.db.scalar(stmt)

    def find_neighbors_batch(
        self,
        anchors: list[dict],
        *,
        tenant_id: str,
        allowed_kb_ids: list[str],
    ) -> list[Chunk]:
        conditions = []
        for anchor in anchors:
            order_index = anchor.get("order_index")
            section_id = anchor.get("section_id")
            if not isinstance(order_index, int) or not section_id:
                continue
            conditions.append(
                and_(
                    Chunk.kb_id == anchor.get("kb_id"),
                    Chunk.document_id == anchor.get("document_id"),
                    Chunk.index_version == anchor.get("index_version"),
                    Chunk.section_id == section_id,
                    Chunk.order_index.in_([order_index - 1, order_index + 1]),
                )
            )
        if not conditions:
            return []
        stmt = select(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.kb_id.in_(allowed_kb_ids),
            or_(*conditions),
        )
        return list(self.db.scalars(stmt).all())

    def first_in_sections_batch(
        self,
        anchors: list[dict],
        *,
        tenant_id: str,
        allowed_kb_ids: list[str],
    ) -> list[Chunk]:
        conditions = []
        for anchor in anchors:
            parent_section_id = anchor.get("parent_section_id")
            if not parent_section_id:
                continue
            conditions.append(
                and_(
                    Chunk.kb_id == anchor.get("kb_id"),
                    Chunk.document_id == anchor.get("document_id"),
                    Chunk.index_version == anchor.get("index_version"),
                    Chunk.section_id == parent_section_id,
                )
            )
        if not conditions:
            return []
        stmt = (
            select(Chunk)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.kb_id.in_(allowed_kb_ids),
                or_(*conditions),
            )
            .order_by(Chunk.order_index)
        )
        chunks = list(self.db.scalars(stmt).all())
        first_by_section: dict[tuple[str, str, str, str], Chunk] = {}
        for chunk in chunks:
            key = (chunk.kb_id, chunk.document_id, chunk.index_version, str(chunk.section_id))
            first_by_section.setdefault(key, chunk)
        return list(first_by_section.values())

    def get_by_ids_in_allowed_scope(
        self,
        chunk_ids: list[str],
        *,
        tenant_id: str,
        allowed_kb_ids: list[str],
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(
            Chunk.id.in_(chunk_ids),
            Chunk.tenant_id == tenant_id,
            Chunk.kb_id.in_(allowed_kb_ids),
        )
        return list(self.db.scalars(stmt).all())

    def get_by_ids_in_versioned_scope(
        self,
        chunk_ids: list[str],
        *,
        tenant_id: str,
        index_versions: dict[str, str],
    ) -> list[Chunk]:
        if not chunk_ids or not index_versions:
            return []
        version_conditions = [
            and_(Chunk.kb_id == kb_id, Chunk.index_version == index_version)
            for kb_id, index_version in index_versions.items()
        ]
        stmt = select(Chunk).where(
            Chunk.id.in_(chunk_ids),
            Chunk.tenant_id == tenant_id,
            or_(*version_conditions),
        )
        return list(self.db.scalars(stmt).all())
