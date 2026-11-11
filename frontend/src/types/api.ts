export interface ApiResponse<T> {
  code: number;
  msg: string;
  data: T;
}

export interface CreateKnowledgeBaseRequest {
  name: string;
  description?: string;
}

export interface KnowledgeBaseData {
  kb_id: string;
  name: string;
  tenant_id: string;
  created_at: string;
}

export interface KnowledgeBaseDetailData {
  kb_id: string;
  tenant_id: string;
  name: string;
  description: string | null;
  settings: Record<string, unknown>;
  active_index_version: string;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface UpdateKnowledgeBaseRequest {
  name?: string;
  description?: string;
  settings?: Record<string, unknown>;
}

export interface DocumentTreeItem {
  document_id: string;
  title: string;
  status: number;
  chunk_count: number;
  error_message?: string | null;
  created_at: string;
}

export interface IndexVersionTreeItem {
  version: string;
  status: "building" | "ready" | "active" | "failed" | "archived";
  is_current: boolean;
  document_count: number;
  chunk_count: number;
  created_at: string;
  completed_at?: string | null;
  documents: DocumentTreeItem[];
}

export interface KnowledgeBaseTreeItem {
  kb_id: string;
  name: string;
  description: string | null;
  active_index_version: string;
  created_at: string;
  documents: DocumentTreeItem[];
  index_versions: IndexVersionTreeItem[];
}

export interface TenantTreeItem {
  tenant_id: string;
  knowledge_bases: KnowledgeBaseTreeItem[];
}

export interface AuthFeatures {
  allowed_profiles: string[];
  hybrid_allowed: boolean;
  rerank_allowed: boolean;
  evidence_allowed: boolean;
  query_rewrite_allowed: boolean;
  research_allowed: boolean;
}

export interface AuthLimits {
  retrieve_qps: number;
  retrieve_daily: number;
  max_kb: number;
  max_kb_per_retrieve: number;
  max_documents_per_kb: number;
  max_processing_documents: number;
}

export interface AuthUsage {
  kb_count: number;
  retrieve_daily: number;
}

export interface AuthMeData {
  tenant_id: string;
  tenant_name: string;
  key_prefix: string | null;
  key_name: string | null;
  plan: string;
  features: AuthFeatures;
  limits: AuthLimits;
  usage: AuthUsage;
}

export interface UploadDocumentRequest {
  kb_id: string;
  title: string;
  content: string;
}

export interface UploadDocumentData {
  document_id: string;
  kb_id: string;
  status: number;
  chunk_count: number;
}

export interface UploadFileResult {
  fileName: string;
  relativePath: string;
  status: "submitted" | "success" | "failed";
  documentId?: string;
  chunkCount?: number;
  message?: string;
}

export type RetrievalMode = "vector" | "bm25" | "hybrid";

export interface RetrievalOptions {
  mode?: RetrievalMode;
  vector_top_k?: number;
  bm25_top_k?: number;
}

export interface RerankOptions {
  enabled?: boolean;
  top_n?: number;
}

export interface EvidenceOptions {
  enabled?: boolean;
  max_items?: number;
}

export interface QueryOptions {
  enabled?: boolean;
  strategy?: "noop" | "rewrite";
  synonym_enabled?: boolean;
}

export type RetrieveProfile = "speed" | "balanced" | "quality" | "custom";

export interface RetrieveRequest {
  kb_id?: string;
  kb_ids?: string[];
  user_id: string;
  query: string;
  index_version?: string;
  profile: RetrieveProfile;
  top_k?: number;
  retrieval_options?: RetrievalOptions;
  rerank_options?: RerankOptions;
  evidence_options?: EvidenceOptions;
  query_options?: QueryOptions;
}

export interface RetrievedChunkData {
  kb_id: string;
  kb_name?: string | null;
  document_id: string;
  chunk_id: string;
  title: string;
  content: string;
  retrieval_text?: string | null;
  index_version?: string;
  section_id?: string | null;
  parent_section_id?: string | null;
  order_index?: number | null;
  score: number;
  vector_score: number | null;
  bm25_score: number | null;
  vector_rank: number | null;
  bm25_rank: number | null;
  retrieval_source: string | null;
  rerank_score: number | null;
  metadata?: Record<string, unknown> | null;
  context?: RetrievedChunkContext | null;
}

export type RetrievedChunkRelation =
  | "anchor"
  | "previous"
  | "next"
  | "same_section"
  | "parent_section"
  | "reference";

export interface RetrievedChunkContext {
  content: string;
  chunk_ids: string[];
  sources: Array<{
    chunk_id: string;
    relation: RetrievedChunkRelation;
    content?: string | null;
    document_id?: string | null;
    title?: string | null;
    metadata?: Record<string, unknown> | null;
  }>;
  latency_ms?: number | null;
  degraded?: boolean;
  error?: string | null;
}

export interface RetrieveRerankMetadata {
  enabled: boolean;
  provider: string;
  llm_provider: string | null;
  model: string | null;
  top_n: number | null;
  candidate_count: number;
  returned_count?: number | null;
  latency_ms?: number | null;
  degraded: boolean;
  error: string | null;
}

export interface RetrieveRetrievalMetadata {
  mode?: string;
  fusion?: string | null;
  fusion_scope?: string | null;
  rrf_k?: number | null;
  vector_store?: string;
  keyword_search?: string | null;
  vector_top_k?: number | null;
  bm25_top_k?: number | null;
  vector_count?: number;
  bm25_count?: number;
  fused_count?: number;
  degraded?: boolean;
  degraded_reason?: string | null;
  cost_ms?: number;
  multi_kb?: boolean;
  kb_count?: number;
  kb_ids?: string[];
  index_version?: string;
  index_versions?: Record<string, string>;
  embedding_request_count?: number;
  vector_request_count?: number;
  bm25_request_count?: number;
  source_distribution?: Record<string, number>;
  vector_latency_ms?: number;
  embedding_latency_ms?: number;
  vector_store_latency_ms?: number;
  bm25_latency_ms?: number;
  hybrid_latency_ms?: number;
}

export interface QueryProcessingMetadata {
  enabled: boolean;
  strategy: string;
  raw_query: string;
  effective_query: string;
  search_query: string;
  latency_ms: number;
  synonym_expansions: string[];
  synonym_enabled: boolean;
  synonym_applied: boolean;
  degraded: boolean;
  degraded_reason: string | null;
}

export interface TenantPolicyMetadata {
  plan: string;
  retrieve_profile: string;
  effective_mode: string;
  effective_rerank: boolean;
  effective_evidence: boolean;
  effective_query_rewrite: boolean;
  rerank_disabled_reason?: string | null;
  evidence_disabled_reason?: string | null;
  query_rewrite_disabled_reason?: string | null;
  max_kb_per_retrieve?: number | null;
  actual_kb_count?: number | null;
}

export interface RetrieveMetadata {
  log_id: string;
  trace_id?: string | null;
  top_k: number;
  vector_store: string;
  latency_ms: number;
  retrieval: RetrieveRetrievalMetadata | null;
  candidate_filter?: {
    candidate_count_before_filter?: number;
    candidate_count_after_filter?: number;
    filtered_count?: number;
    filtered_reasons?: Record<string, number>;
  } | null;
  context?: {
    executed: boolean;
    anchor_count: number;
    expanded_anchor_count: number;
    supplemental_chunk_count: number;
    deduplicated_count: number;
    latency_ms: number;
    degraded: boolean;
    error?: string | null;
  } | null;
  rerank: RetrieveRerankMetadata;
  evidence?: RetrieveEvidenceMetadata;
  query_processing: QueryProcessingMetadata | null;
  tenant_policy?: TenantPolicyMetadata | null;
}

export interface RetrieveData {
  query: string;
  kb_id: string;
  kb_ids: string[];
  retrieved_chunks: RetrievedChunkData[];
  evidence_pack?: EvidencePack | null;
  metadata: RetrieveMetadata;
}

export type EvidenceRole = "core" | "supporting";
export type EvidenceStatus = "complete" | "partial" | "missing";

export interface EvidenceItem {
  evidence_id: string;
  chunk_id: string;
  kb_id: string;
  document_id: string;
  title: string;
  heading_path?: string | null;
  index_version: string;
  content: string;
  source: string;
  retrieved_rank?: number | null;
  rounds?: number[];
  query_ids?: string[];
}

export interface EvidenceGroup {
  aspect: string;
  evidence: Array<{ evidence_id: string; role: EvidenceRole }>;
  covered: boolean;
}

export interface EvidencePack {
  status: EvidenceStatus;
  missing_aspects: string[];
  groups: EvidenceGroup[];
  items: EvidenceItem[];
}

export interface RetrieveEvidenceMetadata {
  enabled: boolean;
  executed: boolean;
  candidate_count: number;
  evidence_count: number;
  output_chars: number;
  budget_exceeded: boolean;
  rerank_enabled: boolean;
  context_sources_included: boolean;
  status?: EvidenceStatus | null;
  latency_ms: number;
  degraded: boolean;
  error?: string | null;
}

export interface FeedbackRequest {
  trace_id: string;
  log_id?: string;
  score: number;
  comment?: string;
}

export interface FeedbackData {
  feedback_id: string;
  trace_id: string;
  log_id?: string | null;
  score: number;
}

export type RetrieveMode = "single" | "research";

export interface ResearchRequest {
  kb_id?: string;
  kb_ids?: string[];
  user_id: string;
  query: string;
}

export interface ResearchAspect {
  aspect_id: string;
  description: string;
}

export interface PlannedQuery {
  query_id: string;
  query: string;
  aspect_ids: string[];
}

export interface ResearchPlanData {
  aspects: ResearchAspect[];
  initial_queries: PlannedQuery[];
  degraded: boolean;
  error?: string | null;
}

export interface ResearchTaskMetadata {
  query_id: string;
  query: string;
  aspect_ids: string[];
  success: boolean;
  chunk_count: number;
  chunk_ids: string[];
  new_chunk_count: number;
  latency_ms: number;
  degraded: boolean;
  error?: string | null;
}

export interface ResearchRoundMetadata {
  round: number;
  purpose: "initial" | "supplement";
  tasks: ResearchTaskMetadata[];
  candidate_chunk_ids: string[];
  candidate_count: number;
  new_chunk_count: number;
  latency_ms: number;
}

export type ResearchStopReason =
  | "evidence_complete"
  | "no_new_evidence"
  | "max_rounds"
  | "duplicate_query"
  | "timeout"
  | "budget_exhausted"
  | "degraded";

export interface ResearchStageUsage {
  stage: "planner" | "decider" | "finalizer";
  role: "fast" | "strong";
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  degraded: boolean;
}

export interface ResearchMetadata {
  research_id: string;
  log_id: string;
  trace_id?: string | null;
  profile: string;
  plan: string;
  index_versions: Record<string, string>;
  round_count: number;
  retrieval_task_count: number;
  llm_call_count: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  initial_evidence_status: EvidenceStatus;
  stop_reason: ResearchStopReason;
  degraded: boolean;
  errors: string[];
  stages: ResearchStageUsage[];
}

export interface ResearchData {
  query: string;
  kb_id: string;
  kb_ids: string[];
  answer?: string | null;
  evidence_pack?: EvidencePack | null;
  plan: ResearchPlanData;
  rounds: ResearchRoundMetadata[];
  metadata: ResearchMetadata;
}
