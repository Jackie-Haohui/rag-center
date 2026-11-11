from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import AppError
from app.core.index_versions import DEFAULT_INDEX_VERSION, LEGACY_INDEX_VERSION
from app.services.index_version_service import IndexVersionService
from app.services.rag_service import RagService


def version(version: str, status: str):
    now = datetime.now(UTC)
    return SimpleNamespace(
        version=version,
        status=status,
        document_count=1,
        chunk_count=2,
        error_message=None,
        created_at=now,
        completed_at=now,
        activated_at=now,
    )


def make_service() -> IndexVersionService:
    service = object.__new__(IndexVersionService)
    service.db = MagicMock()
    service.version_repository = MagicMock()
    service.knowledge_base_repository = MagicMock()
    service.document_repository = MagicMock()
    service.chunk_repository = MagicMock()
    service.relation_repository = MagicMock()
    return service


def test_initial_index_defaults_to_v2_and_can_explicitly_use_v1() -> None:
    service = make_service()

    service.create_initial(tenant_id="tenant", kb_id="kb")
    default_item = service.version_repository.create.call_args.args[0]
    assert default_item.version == DEFAULT_INDEX_VERSION == "v2"

    service.create_initial(
        tenant_id="tenant",
        kb_id="legacy-kb",
        version=LEGACY_INDEX_VERSION,
    )
    legacy_item = service.version_repository.create.call_args.args[0]
    assert legacy_item.version == "v1"


def test_next_version_uses_highest_existing_number() -> None:
    service = make_service()
    service.version_repository.list_by_kb.return_value = [
        version("v1", "active"),
        version("v2", "archived"),
        version("preview", "failed"),
    ]
    assert service._next_version("tenant", "kb") == "v3"


def test_ready_version_can_activate_and_old_active_is_archived() -> None:
    service = make_service()
    knowledge_base = SimpleNamespace(active_index_version="v1")
    current = version("v1", "active")
    target = version("v2", "ready")
    service._get_kb_or_raise = MagicMock(return_value=knowledge_base)
    service.version_repository.get.side_effect = [target, current]

    data = service.activate("tenant", "kb", "v2")

    assert current.status == "archived"
    assert target.status == "active"
    assert knowledge_base.active_index_version == "v2"
    assert data.version == "v2"


def test_failed_version_cannot_activate() -> None:
    service = make_service()
    service._get_kb_or_raise = MagicMock(return_value=SimpleNamespace(active_index_version="v1"))
    service.version_repository.get.return_value = version("v2", "failed")
    with pytest.raises(AppError, match="只有已就绪"):
        service.activate("tenant", "kb", "v2")


def test_building_version_blocks_document_mutation() -> None:
    service = make_service()
    service.version_repository.get_building.return_value = version("v2", "building")
    with pytest.raises(AppError, match="暂时不能修改文档"):
        service.ensure_not_building("tenant", "kb")


def test_rag_defaults_to_active_version() -> None:
    service = RagService(MagicMock())
    knowledge_base = SimpleNamespace(id="kb", active_index_version="v1")
    service.index_version_repository = MagicMock()

    service.index_version_repository.get.return_value = version("v1", "active")
    assert service._resolve_index_versions(
        "tenant",
        [knowledge_base],
        requested_version=None,
        index_version_overrides=None,
    ) == {"kb": "v1"}


def test_rag_honors_explicit_version_and_rejects_failed_version() -> None:
    service = RagService(MagicMock())
    knowledge_base = SimpleNamespace(id="kb", active_index_version="v2")
    service.index_version_repository = MagicMock()

    service.index_version_repository.get.return_value = version("v1", "archived")
    assert service._resolve_index_versions(
        "tenant",
        [knowledge_base],
        requested_version="v1",
        index_version_overrides=None,
    ) == {"kb": "v1"}

    service.index_version_repository.get.return_value = version("v2", "failed")
    with pytest.raises(AppError, match="不可查询"):
        service._resolve_index_versions(
            "tenant",
            [knowledge_base],
            requested_version="v1",
            index_version_overrides={"kb": "v2"},
        )
