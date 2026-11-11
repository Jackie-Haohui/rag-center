from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.knowledge_base import KnowledgeBase
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.index_version_repository import IndexVersionRepository


async def run(kb_id: str | None) -> None:
    provider = ElasticsearchKeywordSearchProvider()
    backfill = await provider.backfill_v1_index_version()
    print(
        "Elasticsearch backfill: "
        f"total={backfill['total']} updated={backfill['updated']} "
        f"version_conflicts={backfill['version_conflicts']}"
    )

    errors: list[str] = []
    with SessionLocal() as db:
        stmt = select(KnowledgeBase).order_by(KnowledgeBase.id)
        if kb_id:
            stmt = stmt.where(KnowledgeBase.id == kb_id)
        knowledge_bases = list(db.scalars(stmt).all())
        chunks = ChunkRepository(db)
        versions = IndexVersionRepository(db)
        unversioned_db = int(
            db.scalar(select(func.count()).select_from(Chunk).where(Chunk.index_version.is_(None)))
            or 0
        )
        if unversioned_db:
            errors.append(f"database has {unversioned_db} chunks without index_version")

        for knowledge_base in knowledge_bases:
            db_count = chunks.count_by_kb_id(
                knowledge_base.tenant_id,
                knowledge_base.id,
                index_version="v1",
            )
            es_count = await provider.count_by_version(
                tenant_id=knowledge_base.tenant_id,
                kb_id=knowledge_base.id,
                index_version="v1",
            )
            es_missing = await provider.count_by_version(
                tenant_id=knowledge_base.tenant_id,
                kb_id=knowledge_base.id,
                index_version=None,
            )
            v1 = versions.get(knowledge_base.tenant_id, knowledge_base.id, "v1")
            print(
                f"{knowledge_base.id}: db_v1={db_count} es_v1={es_count} "
                f"es_missing={es_missing} active={knowledge_base.active_index_version}"
            )
            if v1 is None:
                errors.append(f"{knowledge_base.id}: missing v1 version row")
            if knowledge_base.active_index_version == "v1" and (v1 is None or v1.status != "active"):
                errors.append(f"{knowledge_base.id}: v1 is not active")
            if es_missing:
                errors.append(f"{knowledge_base.id}: Elasticsearch has unversioned documents")
            if db_count != es_count:
                errors.append(
                    f"{knowledge_base.id}: DB/Elasticsearch v1 count mismatch "
                    f"({db_count} != {es_count})"
                )

    if errors:
        raise RuntimeError("v1 backfill validation failed:\n- " + "\n- ".join(errors))
    print("v1 backfill validation passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill and validate v1 Elasticsearch fields")
    parser.add_argument("--kb-id")
    args = parser.parse_args()
    asyncio.run(run(args.kb_id))


if __name__ == "__main__":
    main()
