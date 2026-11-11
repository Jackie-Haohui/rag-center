from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.core.index_versions import DEFAULT_INDEX_VERSION

RetrieveProfile = Literal["speed", "balanced", "quality", "custom"]


class RerankOptions(BaseModel):
    enabled: bool | None = None
    top_n: int | None = Field(default=None, ge=1, le=50)


class EvidenceOptions(BaseModel):
    enabled: bool | None = None
    max_items: int | None = Field(default=None, ge=1, le=20)


class RetrievalOptions(BaseModel):
    mode: str | None = Field(default=None, pattern="^(vector|bm25|hybrid)$")
    vector_top_k: int | None = Field(default=None, ge=1, le=50)
    bm25_top_k: int | None = Field(default=None, ge=1, le=50)
    rrf_k: int | None = Field(default=None, ge=1)


class QueryOptions(BaseModel):
    enabled: bool | None = None
    strategy: str | None = Field(default=None, pattern="^(noop|rewrite)$")
    synonym_enabled: bool | None = None


class KnowledgeBaseQueryRequest(BaseModel):
    kb_id: str | None = Field(default=None, min_length=1, max_length=36)
    kb_ids: list[str] | None = Field(default=None, min_length=1)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_kb_scope(self) -> Self:
        if self.kb_ids:
            normalized = [item.strip() for item in self.kb_ids if item and item.strip()]
        elif self.kb_id and self.kb_id.strip():
            normalized = [self.kb_id.strip()]
        else:
            raise ValueError("kb_id 或 kb_ids 必须提供一个")

        if not normalized:
            raise ValueError("kb_ids 不能为空")

        deduped: list[str] = []
        seen: set[str] = set()
        for kb_id in normalized:
            if kb_id in seen:
                continue
            seen.add(kb_id)
            deduped.append(kb_id)

        if len(deduped) > settings.multi_kb_max:
            raise ValueError(f"kb_ids 数量不能超过 {settings.multi_kb_max}")

        self.kb_ids = deduped
        self.kb_id = deduped[0]
        return self

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        stripped = self.query.strip()
        if not stripped:
            raise ValueError("query 不能为空")
        if len(stripped) > settings.query_max_length:
            raise ValueError(f"query 长度不能超过 {settings.query_max_length}")
        self.query = stripped
        return self


class RetrievalScopeRequest(KnowledgeBaseQueryRequest):
    index_version: str | None = Field(default=None, pattern=r"^v[1-9][0-9]*$")
    profile: RetrieveProfile | None = None


class RetrieveRequest(RetrievalScopeRequest):
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_options: RetrievalOptions | None = None
    rerank_options: RerankOptions | None = None
    evidence_options: EvidenceOptions | None = None
    query_options: QueryOptions | None = None
    observability_enabled: bool | None = None


class TenantPolicyMetadata(BaseModel):
    plan: str
    retrieve_profile: str
    effective_mode: str
    effective_rerank: bool
    effective_evidence: bool
    effective_query_rewrite: bool
    rerank_disabled_reason: str | None = None
    evidence_disabled_reason: str | None = None
    query_rewrite_disabled_reason: str | None = None
    max_kb_per_retrieve: int | None = None
    actual_kb_count: int | None = None


class RetrievedChunkData(BaseModel):
    kb_id: str
    kb_name: str | None = None
    document_id: str
    chunk_id: str
    title: str
    content: str
    retrieval_text: str | None = None
    index_version: str = DEFAULT_INDEX_VERSION
    section_id: str | None = None
    parent_section_id: str | None = None
    order_index: int | None = None
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    retrieval_source: str | None = None
    rerank_score: float | None = None
    metadata: dict | None = None
    context: "RetrievedChunkContext | None" = None


class RetrievedChunkContextSource(BaseModel):
    chunk_id: str
    relation: Literal["anchor", "previous", "next", "same_section", "parent_section", "reference"]
    content: str | None = None
    document_id: str | None = None
    title: str | None = None
    metadata: dict | None = None


class RetrievedChunkContext(BaseModel):
    content: str
    chunk_ids: list[str]
    sources: list[RetrievedChunkContextSource]
    latency_ms: int | None = None
    degraded: bool = False
    error: str | None = None


class RetrieveContextMetadata(BaseModel):
    executed: bool = False
    anchor_count: int = 0
    expanded_anchor_count: int = 0
    supplemental_chunk_count: int = 0
    deduplicated_count: int = 0
    latency_ms: int = 0
    degraded: bool = False
    error: str | None = None


class RetrieveRerankMetadata(BaseModel):
    enabled: bool
    provider: str
    llm_provider: str | None = None
    model: str | None = None
    top_n: int | None = None
    candidate_count: int = 0
    returned_count: int | None = None
    latency_ms: int | None = None
    degraded: bool = False
    error: str | None = None


class QueryProcessingMetadata(BaseModel):
    enabled: bool
    strategy: str
    raw_query: str
    effective_query: str
    search_query: str
    latency_ms: int
    synonym_expansions: list[str] = Field(default_factory=list)
    synonym_enabled: bool = False
    synonym_applied: bool = False
    degraded: bool = False
    degraded_reason: str | None = None


class RetrieveGraphInjectionMetadata(BaseModel):
    executed: bool = False
    seed_count: int = 0
    candidate_count: int = 0
    injected_count: int = 0
    latency_ms: int = 0
    degraded: bool = False
    error: str | None = None


EvidenceRole = Literal["core", "supporting"]
EvidenceStatus = Literal["complete", "partial", "missing"]


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    kb_id: str
    document_id: str
    title: str
    heading_path: str | None = None
    index_version: str
    content: str
    source: str
    retrieved_rank: int | None = None
    rounds: list[int] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)


class EvidenceGroupItem(BaseModel):
    evidence_id: str
    role: EvidenceRole


class EvidenceGroup(BaseModel):
    aspect: str
    evidence: list[EvidenceGroupItem]
    covered: bool


class EvidencePack(BaseModel):
    status: EvidenceStatus
    missing_aspects: list[str] = Field(default_factory=list)
    groups: list[EvidenceGroup] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)


class RetrieveEvidenceMetadata(BaseModel):
    enabled: bool = False
    executed: bool = False
    candidate_count: int = 0
    evidence_count: int = 0
    output_chars: int = 0
    budget_exceeded: bool = False
    rerank_enabled: bool = False
    context_sources_included: bool = False
    status: EvidenceStatus | None = None
    latency_ms: int = 0
    degraded: bool = False
    error: str | None = None


class RetrieveMetadata(BaseModel):
    log_id: str
    trace_id: str | None = None
    top_k: int
    vector_store: str
    latency_ms: int
    retrieval: dict | None = None
    candidate_filter: dict | None = None
    graph_injection: RetrieveGraphInjectionMetadata | None = None
    context: RetrieveContextMetadata | None = None
    evidence: RetrieveEvidenceMetadata = Field(default_factory=RetrieveEvidenceMetadata)
    rerank: RetrieveRerankMetadata
    query_processing: QueryProcessingMetadata | None = None
    tenant_policy: TenantPolicyMetadata | None = None


class RetrieveData(BaseModel):
    query: str
    kb_id: str
    kb_ids: list[str]
    retrieved_chunks: list[RetrievedChunkData]
    evidence_pack: EvidencePack | None = None
    metadata: RetrieveMetadata
