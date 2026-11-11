from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.services.index_version_service import IndexVersionService


async def run(tenant_id: str, kb_id: str) -> None:
    with SessionLocal() as db:
        service = IndexVersionService(db)
        data = service.start_rebuild(tenant_id, kb_id)
        print(f"building {data.version}")
        await service.rebuild(tenant_id, kb_id, data.version)
        versions = service.list_versions(tenant_id, kb_id)
        built = next(item for item in versions if item.version == data.version)
        print(
            f"ready {built.version}: documents={built.document_count} chunks={built.chunk_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a contextual index version synchronously")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--kb-id", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.tenant_id, args.kb_id))


if __name__ == "__main__":
    main()
