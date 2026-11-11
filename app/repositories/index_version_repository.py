from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.index_version import KnowledgeBaseIndexVersion


class IndexVersionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, item: KnowledgeBaseIndexVersion) -> KnowledgeBaseIndexVersion:
        self.db.add(item)
        self.db.flush()
        return item

    def get(
        self, tenant_id: str, kb_id: str, version: str
    ) -> KnowledgeBaseIndexVersion | None:
        stmt = select(KnowledgeBaseIndexVersion).where(
            KnowledgeBaseIndexVersion.tenant_id == tenant_id,
            KnowledgeBaseIndexVersion.kb_id == kb_id,
            KnowledgeBaseIndexVersion.version == version,
        )
        return self.db.scalar(stmt)

    def get_building(self, tenant_id: str, kb_id: str) -> KnowledgeBaseIndexVersion | None:
        stmt = select(KnowledgeBaseIndexVersion).where(
            KnowledgeBaseIndexVersion.tenant_id == tenant_id,
            KnowledgeBaseIndexVersion.kb_id == kb_id,
            KnowledgeBaseIndexVersion.status == "building",
        )
        return self.db.scalar(stmt)

    def list_by_kb(self, tenant_id: str, kb_id: str) -> list[KnowledgeBaseIndexVersion]:
        stmt = (
            select(KnowledgeBaseIndexVersion)
            .where(
                KnowledgeBaseIndexVersion.tenant_id == tenant_id,
                KnowledgeBaseIndexVersion.kb_id == kb_id,
            )
            .order_by(KnowledgeBaseIndexVersion.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def delete(self, item: KnowledgeBaseIndexVersion) -> None:
        self.db.delete(item)
        self.db.flush()
