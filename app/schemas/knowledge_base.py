from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.core.index_versions import DEFAULT_INDEX_VERSION


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class UpdateKnowledgeBaseRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    settings: dict | None = None

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "UpdateKnowledgeBaseRequest":
        if self.name is None and self.description is None and self.settings is None:
            raise ValueError("至少提供一个待更新字段")
        return self


class KnowledgeBaseData(BaseModel):
    kb_id: str
    name: str
    tenant_id: str
    created_at: datetime


class KnowledgeBaseDetailData(BaseModel):
    kb_id: str
    tenant_id: str
    name: str
    description: str | None
    settings: dict
    active_index_version: str = DEFAULT_INDEX_VERSION
    document_count: int
    created_at: datetime
    updated_at: datetime


class DocumentTreeItem(BaseModel):
    document_id: str
    title: str
    status: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime


class IndexVersionTreeItem(BaseModel):
    version: str
    status: str
    is_current: bool
    document_count: int
    chunk_count: int
    created_at: datetime
    completed_at: datetime | None = None
    documents: list[DocumentTreeItem]


class KnowledgeBaseTreeItem(BaseModel):
    kb_id: str
    name: str
    description: str | None
    active_index_version: str = DEFAULT_INDEX_VERSION
    created_at: datetime
    documents: list[DocumentTreeItem]
    index_versions: list[IndexVersionTreeItem] = Field(default_factory=list)


class TenantTreeItem(BaseModel):
    tenant_id: str
    knowledge_bases: list[KnowledgeBaseTreeItem]


class IndexVersionData(BaseModel):
    version: str
    status: str
    document_count: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    activated_at: datetime | None = None


class RebuildIndexData(BaseModel):
    kb_id: str
    version: str
    status: str
