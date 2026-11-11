from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.rag import (
    EvidencePack,
    KnowledgeBaseQueryRequest,
)

ResearchStopReason = Literal[
    "evidence_complete",
    "no_new_evidence",
    "max_rounds",
    "duplicate_query",
    "timeout",
    "budget_exhausted",
    "degraded",
]

ResearchStage = Literal[
    "planning",
    "retrieving",
    "deciding",
    "supplementing",
    "finalizing",
    "completed",
]
ResearchProblemType = Literal["simple", "independent", "dependent"]
ResearchAspectStage = Literal["premise", "answer"]


class ResearchAspect(BaseModel):
    aspect_id: str
    description: str
    stage: ResearchAspectStage = "answer"


class PlannedQuery(BaseModel):
    query_id: str
    query: str
    aspect_ids: list[str]


class ResearchCandidate(BaseModel):
    candidate_id: str = ""
    chunk_id: str
    kb_id: str
    document_id: str
    index_version: str
    title: str
    heading_path: str | None = None
    content: str
    retrieval_text: str | None = None
    source: str
    aspect_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    rounds: list[int] = Field(default_factory=list)
    best_rank: int | None = None
    fusion_score: float = 0.0
    context_relation: str | None = None
    metadata: dict = Field(default_factory=dict)


class ResearchTaskResult(BaseModel):
    query_id: str
    query: str
    aspect_ids: list[str]
    round: int
    success: bool
    latency_ms: int
    chunk_ids: list[str] = Field(default_factory=list)
    new_chunk_count: int = 0
    degraded: bool = False
    error: str | None = None


class ResearchRound(BaseModel):
    round: int
    purpose: Literal["initial", "supplement"]
    tasks: list[ResearchTaskResult] = Field(default_factory=list)
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    new_chunk_count: int = 0
    latency_ms: int = 0


class ResearchState(BaseModel):
    research_id: str
    original_query: str
    tenant_id: str
    kb_ids: list[str]
    index_versions: dict[str, str]
    stage: ResearchStage = "planning"
    problem_type: ResearchProblemType = "simple"
    aspects: list[ResearchAspect] = Field(default_factory=list)
    initial_queries: list[PlannedQuery] = Field(default_factory=list)
    missing_aspect_ids: list[str] = Field(default_factory=list)
    rounds: list[ResearchRound] = Field(default_factory=list)
    visited_queries: list[str] = Field(default_factory=list)
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    round_count: int = 0
    retrieval_task_count: int = 0
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    stop_reason: ResearchStopReason | None = None
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)

    def record_usage(self, usage: dict | None) -> None:
        value = usage or {}
        self.input_tokens += int(value.get("prompt_tokens", 0) or 0)
        self.output_tokens += int(value.get("completion_tokens", 0) or 0)

    def has_visited(self, query: str) -> bool:
        return normalize_research_query(query) in self.visited_queries

    def mark_visited(self, query: str) -> None:
        normalized = normalize_research_query(query)
        if normalized and normalized not in self.visited_queries:
            self.visited_queries.append(normalized)


def normalize_research_query(query: str) -> str:
    normalized = " ".join(query.strip().casefold().split())
    return normalized.rstrip("?？。.!！")


class ResearchRequest(KnowledgeBaseQueryRequest):
    model_config = ConfigDict(extra="forbid")


class ResearchPlanData(BaseModel):
    problem_type: ResearchProblemType = "simple"
    aspects: list[ResearchAspect]
    initial_queries: list[PlannedQuery]
    degraded: bool = False
    error: str | None = None


class ResearchTaskMetadata(BaseModel):
    query_id: str
    query: str
    aspect_ids: list[str]
    success: bool
    chunk_count: int
    chunk_ids: list[str] = Field(default_factory=list)
    new_chunk_count: int
    latency_ms: int
    degraded: bool = False
    error: str | None = None


class ResearchRoundMetadata(BaseModel):
    round: int
    purpose: Literal["initial", "supplement"]
    tasks: list[ResearchTaskMetadata]
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    candidate_count: int
    new_chunk_count: int
    latency_ms: int


class ResearchStageUsage(BaseModel):
    stage: Literal["planner", "decider", "finalizer"]
    role: Literal["fast", "strong"]
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    degraded: bool = False


class ResearchMetadata(BaseModel):
    research_id: str
    log_id: str
    trace_id: str | None = None
    profile: str
    plan: str
    index_versions: dict[str, str]
    round_count: int
    retrieval_task_count: int
    llm_call_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    initial_evidence_status: Literal["complete", "partial", "missing"]
    stop_reason: ResearchStopReason
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)
    stages: list[ResearchStageUsage] = Field(default_factory=list)


class ResearchData(BaseModel):
    query: str
    kb_id: str
    kb_ids: list[str]
    # Research always returns these fields. ``answer`` may only be null when the
    # bounded Finalizer explicitly degrades; the evidence pack itself is never
    # omitted.
    answer: str | None
    evidence_pack: EvidencePack
    plan: ResearchPlanData
    rounds: list[ResearchRoundMetadata]
    metadata: ResearchMetadata
