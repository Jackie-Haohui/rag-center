#!/usr/bin/env python3
"""把卡在「处理中」的文档重新投递索引任务。

用法:
    uv run python scripts/requeue_stuck_documents.py

前提: Celery worker 已在运行（--pool=solo）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.document import Document
from app.tasks.indexing import index_document_task
from app.utils.status import DocumentStatus


def main() -> int:
    db = SessionLocal()
    try:
        stmt = select(Document).where(Document.status == DocumentStatus.PROCESSING)
        documents = list(db.scalars(stmt).all())
        if not documents:
            print("没有处于「处理中」的文档，无需重投。")
            return 0

        print(f"找到 {len(documents)} 个卡在「处理中」的文档，重新投递索引任务：")
        for doc in documents:
            index_document_task.delay(doc.id)
            print(f"  queued: {doc.id} | {doc.title}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
