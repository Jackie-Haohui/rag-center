import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, HelpCircle, Loader2, Star } from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { AppHeader } from "@/components/AppHeader";
import { PlanBadge } from "@/components/PlanBadge";
import { ProfilePresetHelpTooltip } from "@/components/ProfilePresetHelpTooltip";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { MarkdownContent } from "@/components/MarkdownContent";
import { cn } from "@/lib/utils";
import { fetchAuthMe } from "@/services/authService";
import { fetchTree } from "@/services/knowledgeBaseService";
import { research, retrieve, submitFeedback } from "@/services/ragService";
import type {
  AuthFeatures,
  EvidencePack,
  KnowledgeBaseTreeItem,
  ResearchData,
  ResearchRequest,
  RetrieveData,
  RetrieveMode,
  RetrieveProfile,
  RetrieveRequest,
  RetrievalMode,
  RetrievedChunkData,
} from "@/types/api";
import {
  getDefaultProfile,
  getPlanLabel,
  getProfileDisabledTooltip,
  isProfileAllowed,
  PROFILE_OPTIONS,
} from "@/utils/tenantPlan";

const RETRIEVAL_MODES: RetrievalMode[] = ["vector", "bm25", "hybrid"];
const INDEX_VERSIONS = ["v2", "v1"] as const;
const QUERYABLE_INDEX_STATUSES = new Set(["active", "ready", "archived"]);
const fieldLabelClass = "w-[100px] shrink-0 text-right text-sm font-medium";

const PARAM_TOOLTIPS = {
  top_k: "最终返回几条最相关的结果。越大召回越多，但也可能掺入不太相关的内容。",
  mode: "vector：按语义相似度搜；bm25：按关键词字面匹配；hybrid：两路都搜再合并，中文文档一般更稳。",
  vector_top_k: "混合模式下，语义检索先捞多少条候选。越大覆盖面越广，速度会稍慢。",
  bm25_top_k: "混合模式下，关键词检索先捞多少条候选。适合含职级、专有名词等字面匹配的问题。",
  rerank: "用专用精排模型对初筛结果重新打分排序；调试初始召回时可先关掉。",
  evidence: "从本次已有候选中选择并分组回答所需证据；不会改变检索结果，也不会生成答案。",
  query_rewrite:
    "用 AI 把口语问题改成更好搜的说法；更慢、消耗 LLM，调试时可对比开/关效果。",
} as const;

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDecimal(value: number, digits = 4) {
  return value.toFixed(digits);
}

interface ChunkMetric {
  label: string;
  value: string;
  description: string;
}

function buildChunkMetrics(chunk: RetrievedChunkData): ChunkMetric[] {
  const metrics: ChunkMetric[] = [
    {
      label: "综合排序分",
      value: formatDecimal(chunk.score),
      description: "检索阶段的排序依据；混合模式下为语义与关键词双路融合分",
    },
  ];

  if (chunk.vector_score != null) {
    metrics.push({
      label: "语义相似度",
      value: formatPercent(chunk.vector_score),
      description: "向量检索：问题与文本在语义上的接近程度",
    });
  }

  if (chunk.bm25_score != null) {
    metrics.push({
      label: "关键词匹配分",
      value: formatDecimal(chunk.bm25_score, 2),
      description: "关键词检索：问题与文本的字面匹配强度",
    });
  }

  if (chunk.rerank_score != null) {
    metrics.push({
      label: "AI 相关性",
      value: formatPercent(chunk.rerank_score),
      description: "重排阶段：模型判断该片段能否有效回答问题",
    });
  }

  if (chunk.vector_rank != null) {
    metrics.push({
      label: "语义检索排名",
      value: `第 ${chunk.vector_rank} 名`,
      description: "在纯语义检索结果中的名次，1 为最高",
    });
  }

  if (chunk.bm25_rank != null) {
    metrics.push({
      label: "关键词检索排名",
      value: `第 ${chunk.bm25_rank} 名`,
      description: "在纯关键词检索结果中的名次，1 为最高",
    });
  }

  return metrics;
}

function buildRetrievePayload(
  selectedKbIds: string[],
  form: Omit<RetrieveForm, "selectedKbIds">,
  profile: RetrieveProfile,
  features: AuthFeatures,
): RetrieveRequest {
  const payload: RetrieveRequest = {
    user_id: "debug_user",
    query: form.query.trim(),
    index_version: form.index_version,
    profile,
  };

  if (selectedKbIds.length === 1) {
    payload.kb_id = selectedKbIds[0];
  } else {
    payload.kb_ids = selectedKbIds;
  }

  if (profile !== "custom") {
    return payload;
  }

  payload.top_k = form.top_k;
  payload.retrieval_options = { mode: form.mode };

  if (form.mode === "hybrid") {
    payload.retrieval_options = {
      mode: form.mode,
      vector_top_k: form.vector_top_k,
      bm25_top_k: form.bm25_top_k,
    };
  }

  const rerankEnabled = form.rerank_enabled && features.rerank_allowed;
  payload.rerank_options = rerankEnabled
    ? { enabled: true, top_n: form.rerank_top_n }
    : { enabled: false };

  const evidenceEnabled = form.evidence_enabled && features.evidence_allowed;
  payload.evidence_options = evidenceEnabled
    ? { enabled: true, max_items: form.evidence_max_items }
    : { enabled: false };

  const queryRewriteEnabled = form.query_rewrite_enabled && features.query_rewrite_allowed;
  payload.query_options = {
    enabled: queryRewriteEnabled,
    strategy: queryRewriteEnabled ? "rewrite" : "noop",
    synonym_enabled: form.synonym_enabled,
  };

  return payload;
}

function buildResearchPayload(
  selectedKbIds: string[],
  form: Omit<RetrieveForm, "selectedKbIds">,
): ResearchRequest {
  const payload: ResearchRequest = {
    user_id: "debug_user",
    query: form.query.trim(),
  };
  if (selectedKbIds.length === 1) {
    payload.kb_id = selectedKbIds[0];
  } else {
    payload.kb_ids = selectedKbIds;
  }
  return payload;
}

interface RetrieveForm {
  query: string;
  index_version: string;
  top_k: number;
  mode: RetrievalMode;
  vector_top_k: number;
  bm25_top_k: number;
  rerank_enabled: boolean;
  rerank_top_n: number;
  evidence_enabled: boolean;
  evidence_max_items: number;
  query_rewrite_enabled: boolean;
  synonym_enabled: boolean;
}

const defaultForm = (indexVersion = "v2"): RetrieveForm => ({
  query: "",
  index_version: indexVersion,
  top_k: 5,
  mode: "hybrid",
  vector_top_k: 20,
  bm25_top_k: 20,
  rerank_enabled: false,
  rerank_top_n: 5,
  evidence_enabled: false,
  evidence_max_items: 8,
  query_rewrite_enabled: false,
  synonym_enabled: false,
});

export function RetrievePage() {
  const [searchParams] = useSearchParams();
  const queryKbId = searchParams.get("kb_id") ?? "";
  const queryIndexVersion = searchParams.get("index_version");
  const initialIndexVersion =
    queryIndexVersion && /^v[1-9][0-9]*$/.test(queryIndexVersion)
      ? queryIndexVersion
      : "v2";

  const { data: tenantInfo, isLoading: authLoading } = useQuery({
    queryKey: ["authMe"],
    queryFn: fetchAuthMe,
  });

  const { data: kbTree, isLoading: treeLoading } = useQuery({
    queryKey: ["kbTree"],
    queryFn: () => fetchTree(),
  });

  const knowledgeBases: KnowledgeBaseTreeItem[] = useMemo(
    () => kbTree?.flatMap((tenant) => tenant.knowledge_bases) ?? [],
    [kbTree],
  );

  const appliedQueryKbRef = useRef<string | null>(null);
  const [form, setForm] = useState<RetrieveForm>(() => defaultForm(initialIndexVersion));
  const [retrieveMode, setRetrieveMode] = useState<RetrieveMode>("single");
  const [selectedKbIds, setSelectedKbIds] = useState<string[]>([]);
  const [profile, setProfile] = useState<RetrieveProfile>("balanced");
  const [paramsExpanded, setParamsExpanded] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [feedbackScore, setFeedbackScore] = useState<number | null>(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!queryKbId || knowledgeBases.length === 0) {
      return;
    }
    if (appliedQueryKbRef.current === queryKbId) {
      return;
    }
    if (knowledgeBases.some((kb) => kb.kb_id === queryKbId)) {
      setSelectedKbIds([queryKbId]);
      appliedQueryKbRef.current = queryKbId;
    }
  }, [queryKbId, knowledgeBases]);

  useEffect(() => {
    if (!tenantInfo) {
      return;
    }
    const nextProfile = getDefaultProfile(tenantInfo.features);
    setProfile((current) =>
      isProfileAllowed(tenantInfo.features, current) ? current : nextProfile,
    );
    setParamsExpanded(false);
    if (!tenantInfo.features.hybrid_allowed) {
      setForm((prev) => ({ ...prev, mode: "vector" }));
    }
  }, [tenantInfo]);

  useEffect(() => {
    setParamsExpanded(profile === "custom");
  }, [profile]);

  const retrieveMutation = useMutation({
    mutationFn: retrieve,
  });

  const researchMutation = useMutation({
    mutationFn: research,
  });

  const feedbackMutation = useMutation({
    mutationFn: submitFeedback,
    onSuccess: (data) => {
      setFeedbackSubmitted(true);
      setFeedbackMessage(`已提交，已评 ${data.score} 分`);
    },
    onError: (error) => {
      setFeedbackMessage((error as Error).message);
    },
  });

  function updateForm<K extends keyof RetrieveForm>(key: K, value: RetrieveForm[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function handleProfileChange(nextProfile: RetrieveProfile) {
    if (!tenantInfo || !isProfileAllowed(tenantInfo.features, nextProfile)) {
      return;
    }
    setProfile(nextProfile);
  }

  function handleRetrieveModeChange(nextMode: RetrieveMode) {
    if (nextMode === "research" && !tenantInfo?.features.research_allowed) {
      return;
    }
    setRetrieveMode(nextMode);
    retrieveMutation.reset();
    researchMutation.reset();
    setFormError(null);
    setFeedbackScore(null);
    setFeedbackComment("");
    setFeedbackSubmitted(false);
    setFeedbackMessage(null);
  }

  function toggleKbSelection(kbId: string) {
    setSelectedKbIds((current) => {
      if (current.includes(kbId)) {
        return current.filter((id) => id !== kbId);
      }
      const maxKb = tenantInfo?.limits.max_kb_per_retrieve ?? 1;
      if (current.length >= maxKb) {
        setFormError(`当前套餐最多支持 ${maxKb} 个知识库联合检索`);
        return current;
      }
      setFormError(null);
      return [...current, kbId];
    });
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!tenantInfo) {
      setFormError("正在加载租户套餐信息，请稍候");
      return;
    }
    if (selectedKbIds.length === 0 || !form.query.trim()) {
      setFormError("请至少选择一个知识库并填写 query");
      return;
    }
    if (selectedKbIds.length > tenantInfo.limits.max_kb_per_retrieve) {
      setFormError(`当前套餐最多支持 ${tenantInfo.limits.max_kb_per_retrieve} 个知识库联合检索`);
      return;
    }
    const requestedIndexVersion = retrieveMode === "research" ? "v2" : form.index_version;
    const unavailableKb = knowledgeBases.find(
      (kb) =>
        selectedKbIds.includes(kb.kb_id) &&
        !kb.index_versions.some(
          (item) =>
            item.version === requestedIndexVersion && QUERYABLE_INDEX_STATUSES.has(item.status),
        ),
    );
    if (unavailableKb) {
      setFormError(`知识库「${unavailableKb.name}」没有可查询的 ${requestedIndexVersion} 索引`);
      return;
    }
    setFormError(null);
    setFeedbackScore(null);
    setFeedbackComment("");
    setFeedbackSubmitted(false);
    setFeedbackMessage(null);
    if (retrieveMode === "research") {
      researchMutation.mutate(
        buildResearchPayload(selectedKbIds, form),
      );
    } else {
      retrieveMutation.mutate(
        buildRetrievePayload(selectedKbIds, form, profile, tenantInfo.features),
      );
    }
  }

  function handleFeedbackSubmit() {
    if (!activeResult?.metadata.trace_id) {
      return;
    }
    if (feedbackScore === null) {
      setFeedbackMessage("请先选择星级");
      return;
    }
    setFeedbackMessage(null);
    feedbackMutation.mutate({
      trace_id: activeResult.metadata.trace_id,
      log_id: activeResult.metadata.log_id,
      score: feedbackScore,
      comment: feedbackComment.trim() || undefined,
    });
  }

  const result = retrieveMutation.data;
  const researchResult = researchMutation.data;
  const activeResult = retrieveMode === "research" ? researchResult : result;
  const activeMutation = retrieveMode === "research" ? researchMutation : retrieveMutation;
  const features = tenantInfo?.features;
  const showAdvanced = retrieveMode === "single" && profile === "custom";
  const allowedModes: RetrievalMode[] = features?.hybrid_allowed
    ? RETRIEVAL_MODES
    : ["vector"];

  return (
    <main className="min-h-screen px-6 py-6">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <AppHeader />

        <Card>
          <CardHeader>
            <CardTitle>检索模式</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant={retrieveMode === "single" ? "default" : "outline"}
              onClick={() => handleRetrieveModeChange("single")}
            >
              普通检索
            </Button>
            <Button
              type="button"
              variant={retrieveMode === "research" ? "default" : "outline"}
              disabled={!tenantInfo?.features.research_allowed}
              title={
                tenantInfo?.features.research_allowed
                  ? "查询分解、并行检索与最多一次多跳补证"
                  : "当前套餐不支持 Research，请使用专业套餐"
              }
              onClick={() => handleRetrieveModeChange("research")}
            >
              Research（多要点 / 多跳）
            </Button>
            <p className="w-full text-xs text-muted-foreground">
              {retrieveMode === "single"
                ? "一次检索，适合简单事实问题。"
                : "受控执行查询规划、并行检索、证据判断和最多一次补证。"}
            </p>
          </CardContent>
        </Card>

        {retrieveMode === "single" ? (
        <Card>
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <CardTitle>检索 profile</CardTitle>
                <ProfilePresetHelpTooltip />
              </div>
              {tenantInfo ? (
                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  <span>当前套餐：</span>
                  <PlanBadge plan={tenantInfo.plan} showCode />
                </div>
              ) : null}
            </div>
            {authLoading ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                加载套餐信息...
              </div>
            ) : null}
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              {PROFILE_OPTIONS.map((option) => {
                const allowed = tenantInfo
                  ? isProfileAllowed(tenantInfo.features, option.value)
                  : false;
                const tooltip = tenantInfo
                  ? getProfileDisabledTooltip(tenantInfo.plan, option.value)
                  : "";
                return (
                  <ProfileOptionButton
                    key={option.value}
                    label={option.label}
                    description={option.description}
                    selected={profile === option.value}
                    disabled={!allowed}
                    tooltip={allowed ? option.description : tooltip}
                    onClick={() => handleProfileChange(option.value)}
                  />
                );
              })}
            </div>
            {profile !== "custom" ? (
              <p className="text-xs text-muted-foreground">
                已选「{PROFILE_OPTIONS.find((item) => item.value === profile)?.label}
                」，检索参数由后端 preset 展开，无需手动配置。
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                自定义模式下可调整高级参数，仍受当前套餐能力限制。
              </p>
            )}
          </CardContent>
        </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>Research固定配置</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <div className="flex flex-wrap gap-2">
                <Badge>v2索引</Badge>
                <Badge>Hybrid Top20</Badge>
                <Badge>Qwen3精排 Top10</Badge>
                <Badge>最多两轮检索</Badge>
                <Badge>最多10条证据</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Research使用服务端固定配置，不受速度、均衡、质量或自定义Profile影响；
                Query Rewrite和同义词扩展固定关闭。
              </p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>检索配置</CardTitle>
            {showAdvanced ? (
              <Button
                type="button"
                variant="ghost"
                className="h-8 px-2"
                onClick={() => setParamsExpanded((value) => !value)}
              >
                {paramsExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
            ) : null}
          </CardHeader>
          {paramsExpanded || !showAdvanced ? (
            <CardContent className="flex flex-col gap-4">
              <div className="rounded-md border border-border bg-muted/30 px-3 py-3">
                <p className="mb-2 text-xs font-medium text-muted-foreground">检索范围</p>
                {treeLoading ? (
                  <p className="text-sm text-muted-foreground">加载知识库列表...</p>
                ) : knowledgeBases.length === 0 ? (
                  <p className="text-sm text-muted-foreground">当前租户暂无知识库</p>
                ) : (
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {knowledgeBases.map((kb) => (
                      <label
                        key={kb.kb_id}
                        className="flex cursor-pointer items-center gap-2 text-sm"
                      >
                        <input
                          type="checkbox"
                          className="h-4 w-4 shrink-0"
                          checked={selectedKbIds.includes(kb.kb_id)}
                          onChange={() => toggleKbSelection(kb.kb_id)}
                        />
                        <span className="font-medium leading-snug">{kb.name}</span>
                      </label>
                    ))}
                  </div>
                )}
                {tenantInfo ? (
                  <p className="mt-2 text-xs text-muted-foreground">
                    已选 {selectedKbIds.length} / {knowledgeBases.length} 个库
                    （套餐最多可选 {tenantInfo.limits.max_kb_per_retrieve} 个）
                  </p>
                ) : null}
                {retrieveMode === "single" ? (
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
                  <span className="text-xs font-medium text-muted-foreground">索引版本</span>
                  {INDEX_VERSIONS.map((version) => {
                    const unavailable = selectedKbIds.some((kbId) => {
                      const kb = knowledgeBases.find((item) => item.kb_id === kbId);
                      return (
                        kb != null &&
                        !kb.index_versions.some(
                          (item) =>
                            item.version === version &&
                            QUERYABLE_INDEX_STATUSES.has(item.status),
                        )
                      );
                    });
                    return (
                      <Button
                        key={version}
                        type="button"
                        className="h-8"
                        variant={form.index_version === version ? "default" : "outline"}
                        disabled={unavailable}
                        onClick={() => updateForm("index_version", version)}
                      >
                        {version}
                        {version === "v2" ? "（默认）" : ""}
                      </Button>
                    );
                  })}
                  <span className="text-xs text-muted-foreground">
                    当前检索将只查询 {form.index_version}，不会混合不同版本
                  </span>
                </div>
                ) : (
                  <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
                    <span className="text-xs font-medium text-muted-foreground">索引版本</span>
                    <Badge>v2（Research固定）</Badge>
                  </div>
                )}
              </div>

              {showAdvanced ? (
              <div className="flex flex-col gap-3 border-t border-border pt-3">
                <p className="text-xs font-medium text-muted-foreground">高级参数（custom）</p>

              <InlineField label="top_k" htmlFor="retrieve-top-k" tooltip={PARAM_TOOLTIPS.top_k}>
                <Input
                  id="retrieve-top-k"
                  type="number"
                  min={1}
                  max={50}
                  className="h-9 w-20"
                  value={form.top_k}
                  onChange={(event) => updateForm("top_k", Number(event.target.value))}
                />
              </InlineField>

              <div className="flex flex-wrap items-center gap-2">
                <FieldLabel tooltip={PARAM_TOOLTIPS.mode}>检索模式</FieldLabel>
                <div className="flex flex-wrap items-center gap-2">
                  {allowedModes.map((mode) => (
                    <Button
                      key={mode}
                      type="button"
                      className="h-8"
                      variant={form.mode === mode ? "default" : "outline"}
                      onClick={() => updateForm("mode", mode)}
                    >
                      {mode}
                    </Button>
                  ))}
                </div>
              </div>

              {form.mode === "hybrid" ? (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                  <InlineField
                    label="vector_top_k"
                    htmlFor="retrieve-vector-top-k"
                    tooltip={PARAM_TOOLTIPS.vector_top_k}
                  >
                    <Input
                      id="retrieve-vector-top-k"
                      type="number"
                      min={1}
                      max={50}
                      className="h-9 w-20"
                      value={form.vector_top_k}
                      onChange={(event) =>
                        updateForm("vector_top_k", Number(event.target.value))
                      }
                    />
                  </InlineField>
                  <InlineField
                    label="bm25_top_k"
                    htmlFor="retrieve-bm25-top-k"
                    tooltip={PARAM_TOOLTIPS.bm25_top_k}
                  >
                    <Input
                      id="retrieve-bm25-top-k"
                      type="number"
                      min={1}
                      max={50}
                      className="h-9 w-20"
                      value={form.bm25_top_k}
                      onChange={(event) =>
                        updateForm("bm25_top_k", Number(event.target.value))
                      }
                    />
                  </InlineField>
                </div>
              ) : null}

              {retrieveMode === "single" ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <InlineField label="query 改写" tooltip={PARAM_TOOLTIPS.query_rewrite}>
                  <label
                    className={cn(
                      "flex items-center gap-2 text-sm",
                      !features?.query_rewrite_allowed && "cursor-not-allowed opacity-50",
                    )}
                    title={
                      !features?.query_rewrite_allowed
                        ? getProfileDisabledTooltip(tenantInfo?.plan ?? "", "quality")
                        : undefined
                    }
                  >
                    <input
                      type="checkbox"
                      checked={form.query_rewrite_enabled}
                      disabled={!features?.query_rewrite_allowed}
                      onChange={(event) =>
                        updateForm("query_rewrite_enabled", event.target.checked)
                      }
                    />
                    启用
                  </label>
                </InlineField>
                <InlineField label="同义词扩展">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={form.synonym_enabled}
                      onChange={(event) => updateForm("synonym_enabled", event.target.checked)}
                    />
                    启用
                  </label>
                </InlineField>
              </div>
              ) : null}

              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <InlineField label="rerank" tooltip={PARAM_TOOLTIPS.rerank}>
                  <label
                    className={cn(
                      "flex items-center gap-2 text-sm",
                      !features?.rerank_allowed && "cursor-not-allowed opacity-50",
                    )}
                    title={
                      !features?.rerank_allowed
                        ? getProfileDisabledTooltip(tenantInfo?.plan ?? "", "quality")
                        : undefined
                    }
                  >
                    <input
                      type="checkbox"
                      checked={form.rerank_enabled}
                      disabled={!features?.rerank_allowed}
                      onChange={(event) => updateForm("rerank_enabled", event.target.checked)}
                    />
                    启用
                  </label>
                </InlineField>
                {form.rerank_enabled && features?.rerank_allowed ? (
                  <InlineField label="rerank top_n" htmlFor="retrieve-rerank-top-n">
                    <Input
                      id="retrieve-rerank-top-n"
                      type="number"
                      min={1}
                      max={50}
                      className="h-9 w-20"
                      value={form.rerank_top_n}
                      onChange={(event) =>
                        updateForm("rerank_top_n", Number(event.target.value))
                      }
                    />
                  </InlineField>
                ) : null}
              </div>
              {retrieveMode === "single" ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <InlineField label="证据编排" tooltip={PARAM_TOOLTIPS.evidence}>
                  <label
                    className={cn(
                      "flex items-center gap-2 text-sm",
                      !features?.evidence_allowed && "cursor-not-allowed opacity-50",
                    )}
                    title={
                      !features?.evidence_allowed
                        ? getProfileDisabledTooltip(tenantInfo?.plan ?? "", "quality")
                        : undefined
                    }
                  >
                    <input
                      type="checkbox"
                      checked={form.evidence_enabled}
                      disabled={!features?.evidence_allowed}
                      onChange={(event) =>
                        updateForm("evidence_enabled", event.target.checked)
                      }
                    />
                    启用
                  </label>
                </InlineField>
                {form.evidence_enabled && features?.evidence_allowed ? (
                  <InlineField label="最大证据数" htmlFor="retrieve-evidence-max-items">
                    <Input
                      id="retrieve-evidence-max-items"
                      type="number"
                      min={1}
                      max={20}
                      className="h-9 w-20"
                      value={form.evidence_max_items}
                      onChange={(event) =>
                        updateForm("evidence_max_items", Number(event.target.value))
                      }
                    />
                  </InlineField>
                ) : null}
              </div>
              ) : null}
              </div>
              ) : null}
            </CardContent>
          ) : null}
        </Card>

        <form onSubmit={handleSubmit}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>检索与召回</CardTitle>
              {activeResult ? (
                <Badge>
                  {retrieveMode === "research"
                    ? `${researchResult?.evidence_pack?.items.length ?? 0} 条证据`
                    : `${result?.retrieved_chunks.length ?? 0} 条`}
                </Badge>
              ) : null}
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <InlineField label="query" htmlFor="retrieve-query" className="items-start">
                <Textarea
                  id="retrieve-query"
                  rows={4}
                  className="min-w-[320px] flex-1"
                  placeholder="输入检索问题"
                  value={form.query}
                  onChange={(event) => updateForm("query", event.target.value)}
                />
              </InlineField>
              {formError ? <p className="text-sm text-destructive">{formError}</p> : null}
              {activeMutation.error ? (
                <p className="text-sm text-destructive">
                  {(activeMutation.error as Error).message}
                </p>
              ) : null}
              <div>
                <Button type="submit" disabled={activeMutation.isPending}>
                  {activeMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : null}
                  {retrieveMode === "research" ? "开始 Research" : "检索"}
                </Button>
              </div>

              {activeMutation.isPending ? (
                <div className="flex items-center py-6 text-sm text-muted-foreground">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {retrieveMode === "research" ? "Research 执行中..." : "检索中..."}
                </div>
              ) : null}

              {activeResult && !activeMutation.isPending ? (
                <RetrieveFeedbackBar
                  traceId={activeResult.metadata.trace_id}
                  score={feedbackScore}
                  comment={feedbackComment}
                  submitted={feedbackSubmitted}
                  message={feedbackMessage}
                  isSubmitting={feedbackMutation.isPending}
                  onScoreChange={(value) => {
                    setFeedbackScore(value);
                    setFeedbackSubmitted(false);
                    setFeedbackMessage(null);
                  }}
                  onCommentChange={setFeedbackComment}
                  onSubmit={handleFeedbackSubmit}
                />
              ) : null}

              {retrieveMode === "single" && result ? (
                <RetrieveResultsList chunks={result.retrieved_chunks} />
              ) : null}
              {retrieveMode === "single" && result ? (
                <EvidencePackCard
                  pack={result.evidence_pack ?? null}
                  metadata={result.metadata.evidence}
                />
              ) : null}
              {retrieveMode === "research" && researchResult ? (
                <ResearchResults data={researchResult} />
              ) : null}
            </CardContent>
          </Card>
        </form>

        {retrieveMode === "single" && result ? <RetrieveSummaryCard data={result} /> : null}
      </div>
    </main>
  );
}

function ProfileOptionButton({
  label,
  description,
  selected,
  disabled,
  tooltip,
  onClick,
}: {
  label: string;
  description: string;
  selected: boolean;
  disabled: boolean;
  tooltip: string;
  onClick: () => void;
}) {
  return (
    <span className="group/profile relative inline-flex">
      <Button
        type="button"
        variant={selected ? "default" : "outline"}
        className={cn("h-auto flex-col items-start px-3 py-2", disabled && "opacity-50")}
        disabled={disabled}
        onClick={onClick}
        title={tooltip}
      >
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs font-normal opacity-80">{description}</span>
      </Button>
      {disabled ? (
        <span
          role="tooltip"
          className="pointer-events-none absolute left-0 top-full z-20 mt-1 hidden w-64 rounded-md border border-border bg-background px-2.5 py-1.5 text-left text-[11px] font-normal leading-relaxed text-foreground shadow-md group-hover/profile:block"
        >
          {tooltip}
        </span>
      ) : null}
    </span>
  );
}

function InlineField({
  label,
  htmlFor,
  className,
  tooltip,
  children,
}: {
  label: string;
  htmlFor?: string;
  className?: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <FieldLabel htmlFor={htmlFor} tooltip={tooltip}>
        {label}
      </FieldLabel>
      {children}
    </div>
  );
}

function FieldLabel({
  children,
  htmlFor,
  tooltip,
}: {
  children: React.ReactNode;
  htmlFor?: string;
  tooltip?: string;
}) {
  return (
    <div className={cn("flex shrink-0 items-center justify-end gap-1", fieldLabelClass)}>
      <Label htmlFor={htmlFor} className="font-medium">
        {children}
      </Label>
      {tooltip ? <HelpTooltip description={tooltip} /> : null}
    </div>
  );
}

function HelpTooltip({ description }: { description: string }) {
  return (
    <span className="group/tip relative inline-flex shrink-0">
      <HelpCircle
        className="h-3.5 w-3.5 cursor-help text-muted-foreground/60 hover:text-muted-foreground"
        aria-label={description}
      />
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-20 mt-1.5 hidden w-52 -translate-x-1/2 rounded-md border border-border bg-background px-2.5 py-1.5 text-left text-[11px] font-normal leading-relaxed text-foreground shadow-md group-hover/tip:block"
      >
        {description}
      </span>
    </span>
  );
}

function MetricRow({ label, value, description }: ChunkMetric) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <div className="flex min-w-0 items-center gap-1 text-muted-foreground">
        <span className="truncate">{label}</span>
        <HelpTooltip description={description} />
      </div>
      <span className="shrink-0 font-medium tabular-nums">{value}</span>
    </div>
  );
}

function IdLine({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="flex items-start gap-1 break-all text-xs text-muted-foreground">
      <span className="inline-flex shrink-0 items-center gap-1">
        {label}
        <HelpTooltip description={description} />
      </span>
      <span className="font-mono text-[11px] text-muted-foreground/90">{value}</span>
    </div>
  );
}

function ChunkIds({ documentId, chunkId }: { documentId: string; chunkId: string }) {
  return (
    <div className="space-y-1.5 border-t border-border pt-2">
      <IdLine
        label="文档 ID"
        value={documentId}
        description="来源文档的唯一标识，对应知识库中的一篇完整文档"
      />
      <IdLine
        label="切片 ID"
        value={chunkId}
        description="文档切分后的文本块唯一标识，召回内容来自该块"
      />
    </div>
  );
}

function ChunkMetricsList({ metrics }: { metrics: ChunkMetric[] }) {
  return (
    <div className="space-y-2 rounded-md bg-muted/40 px-2.5 py-2">
      {metrics.map((metric) => (
        <MetricRow key={metric.label} {...metric} />
      ))}
    </div>
  );
}

function RetrieveFeedbackBar({
  traceId,
  score,
  comment,
  submitted,
  message,
  isSubmitting,
  onScoreChange,
  onCommentChange,
  onSubmit,
}: {
  traceId?: string | null;
  score: number | null;
  comment: string;
  submitted: boolean;
  message: string | null;
  isSubmitting: boolean;
  onScoreChange: (value: number) => void;
  onCommentChange: (value: string) => void;
  onSubmit: () => void;
}) {
  if (!traceId) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-2 text-sm text-muted-foreground">
        Langfuse 未启用，无法提交检索反馈
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 border-t border-border pt-4">
      <div className="text-sm font-medium">反馈</div>
      <StarRating value={score} onChange={onScoreChange} />
      <Textarea
        rows={2}
        className="min-w-[280px] max-w-xl"
        placeholder="备注（可选），例如：排第三的 chunk 才是对的…"
        value={comment}
        onChange={(event) => onCommentChange(event.target.value)}
      />
      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" variant="outline" disabled={isSubmitting} onClick={onSubmit}>
          {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
          提交反馈
        </Button>
        {submitted && score !== null ? (
          <span className="text-sm text-muted-foreground">已评 {score} 分</span>
        ) : null}
        {message ? (
          <span
            className={cn(
              "text-sm",
              message.startsWith("已提交") ? "text-green-700" : "text-destructive",
            )}
          >
            {message}
          </span>
        ) : null}
      </div>
    </div>
  );
}

const researchStopReasonLabels: Record<ResearchData["metadata"]["stop_reason"], string> = {
  evidence_complete: "首轮证据已完整",
  no_new_evidence: "补证未发现新证据",
  max_rounds: "已达到两轮检索上限",
  duplicate_query: "补证查询与已执行查询重复",
  timeout: "达到 Research 时间上限",
  budget_exhausted: "达到受控预算上限",
  degraded: "链路降级停止",
};

function ResearchResults({ data }: { data: ResearchData }) {
  const answerWithLinks = (data.answer ?? "").replace(
    /\[(E\d+)\]/g,
    "[$1](#evidence-$1)",
  );
  return (
    <div className="space-y-4 border-t border-border pt-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>1. 查询计划</CardTitle>
          {data.plan.degraded ? <Badge>Planner 已降级</Badge> : <Badge>Planner 正常</Badge>}
        </CardHeader>
        <CardContent className="space-y-3">
          {data.plan.degraded && data.plan.error ? (
            <p className="text-sm text-amber-700">{data.plan.error}</p>
          ) : null}
          <div className="grid gap-2 md:grid-cols-2">
            {data.plan.aspects.map((aspect) => (
              <div key={aspect.aspect_id} className="rounded-md border p-3 text-sm">
                <Badge>{aspect.aspect_id}</Badge>
                <span className="ml-2">{aspect.description}</span>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {data.plan.initial_queries.map((query) => (
              <div key={query.query_id} className="rounded-md bg-muted/40 p-3 text-sm">
                <div className="mb-1 flex flex-wrap gap-2">
                  <Badge>{query.query_id}</Badge>
                  {query.aspect_ids.map((aspectId) => (
                    <Badge key={aspectId}>{aspectId}</Badge>
                  ))}
                </div>
                <p>{query.query}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2. 执行时间线</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {data.rounds.map((round) => (
            <div key={round.round} className="rounded-md border p-3">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="font-medium">
                  第 {round.round} 轮：
                  {round.purpose === "initial" ? "并行检索" : "缺失要点补证"}
                </span>
                <Badge>候选 {round.candidate_count}</Badge>
                <Badge>新增 {round.new_chunk_count}</Badge>
                <Badge>{round.latency_ms} ms</Badge>
              </div>
              <div className="space-y-2">
                {round.tasks.map((task) => (
                  <div key={task.query_id} className="rounded bg-muted/40 p-2 text-sm">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge>{task.query_id}</Badge>
                      {task.aspect_ids.map((aspectId) => (
                        <Badge key={aspectId}>{aspectId}</Badge>
                      ))}
                      <span>{task.success ? "成功" : "失败"}</span>
                      <span>命中 {task.chunk_count}</span>
                      <span>新增 {task.new_chunk_count}</span>
                      <span>{task.latency_ms} ms</span>
                      {task.degraded ? <span className="text-amber-700">已降级</span> : null}
                    </div>
                    <p className="mt-1 text-muted-foreground">{task.query}</p>
                    {task.error ? <p className="text-destructive">{task.error}</p> : null}
                  </div>
                ))}
              </div>
            </div>
          ))}
          {data.rounds.length === 1 ? (
            <p className="text-sm text-muted-foreground">
              {data.metadata.stop_reason === "evidence_complete"
                ? "首轮证据已完整，未执行补证。"
                : "本次未执行第二轮补证。"}
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>3. 最终回答</CardTitle>
        </CardHeader>
        <CardContent>
          {data.answer ? (
            <MarkdownContent content={answerWithLinks} openLinksInNewTab={false} />
          ) : (
            <p className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
              回答生成失败或证据不足，可继续查看已获得证据。
            </p>
          )}
        </CardContent>
      </Card>

      <ResearchEvidencePackCard pack={data.evidence_pack ?? null} />

      <Card>
        <CardHeader>
          <CardTitle>5. 用量与停止信息</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="flex flex-wrap gap-2">
            <Badge>轮数 {data.metadata.round_count}</Badge>
            <Badge>检索任务 {data.metadata.retrieval_task_count}</Badge>
            <Badge>LLM {data.metadata.llm_call_count} 次</Badge>
            <Badge>输入 {data.metadata.input_tokens} Token</Badge>
            <Badge>输出 {data.metadata.output_tokens} Token</Badge>
            <Badge>总耗时 {data.metadata.latency_ms} ms</Badge>
          </div>
          <p>
            停止原因：
            <span className="font-medium">
              {researchStopReasonLabels[data.metadata.stop_reason]}
            </span>
            <span className="ml-2 font-mono text-xs text-muted-foreground">
              {data.metadata.stop_reason}
            </span>
          </p>
          <p>运行状态：{data.metadata.degraded ? "已降级" : "正常"}</p>
          {data.metadata.errors.length ? (
            <p className="text-amber-700">{data.metadata.errors.join("；")}</p>
          ) : null}
          <p className="break-all text-xs text-muted-foreground">
            trace_id：{data.metadata.trace_id ?? "未启用 Langfuse"}
          </p>
          <div className="grid gap-2 md:grid-cols-3">
            {data.metadata.stages.map((stage) => (
              <div key={stage.stage} className="rounded-md border p-2 text-xs">
                <p className="font-medium">
                  {stage.stage} / {stage.role}
                </p>
                <p className="truncate text-muted-foreground">{stage.model}</p>
                <p>
                  {stage.latency_ms} ms · {stage.input_tokens}/{stage.output_tokens} Token
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ResearchEvidencePackCard({ pack }: { pack: EvidencePack | null }) {
  if (!pack) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>4. 证据包</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">没有可用证据。</CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>4. 证据包</CardTitle>
        <Badge>{pack.status}</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {pack.missing_aspects.length ? (
          <p className="text-sm text-amber-700">
            缺失要点：{pack.missing_aspects.join("、")}
          </p>
        ) : null}
        {pack.groups.map((group) => (
          <div key={group.aspect} className="rounded-md border p-3 text-sm">
            <span className="font-medium">{group.aspect}</span>
            <Badge className="ml-2">{group.covered ? "已覆盖" : "未覆盖"}</Badge>
            <div className="mt-2 flex flex-wrap gap-2">
              {group.evidence.map((reference) => (
                <a
                  key={`${group.aspect}-${reference.evidence_id}`}
                  href={`#evidence-${reference.evidence_id}`}
                  className="text-primary hover:underline"
                >
                  {reference.evidence_id} · {reference.role}
                </a>
              ))}
            </div>
          </div>
        ))}
        <div className="space-y-3">
          {pack.items.map((item) => (
            <div
              id={`evidence-${item.evidence_id}`}
              key={item.evidence_id}
              className="scroll-mt-6 rounded-md border bg-muted/20 p-3"
            >
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                <Badge>{item.evidence_id}</Badge>
                <span>{item.title}</span>
                {item.heading_path ? <span>{item.heading_path}</span> : null}
                <span>{item.source}</span>
                {item.rounds?.length ? <span>第 {item.rounds.join("、")} 轮</span> : null}
                {item.query_ids?.length ? <span>{item.query_ids.join("、")}</span> : null}
              </div>
              <MarkdownContent content={item.content} />
              <p className="mt-2 break-all font-mono text-[11px] text-muted-foreground">
                Chunk {item.chunk_id} · 索引 {item.index_version}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function StarRating({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => {
        const active = value !== null && star <= value;
        return (
          <button
            key={star}
            type="button"
            aria-label={`${star} 星`}
            className="rounded p-0.5 transition-colors hover:bg-muted"
            onClick={() => onChange(star)}
          >
            <Star
              className={cn(
                "h-6 w-6",
                active ? "fill-amber-400 text-amber-400" : "text-muted-foreground",
              )}
            />
          </button>
        );
      })}
    </div>
  );
}

function RetrieveResultsList({ chunks }: { chunks: RetrievedChunkData[] }) {
  if (chunks.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border py-10 text-center text-sm text-muted-foreground">
        未召回到 chunk
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 border-t border-border pt-4">
      <div className="text-sm font-medium text-muted-foreground">召回结果</div>
      {chunks.map((chunk, index) => (
        <div
          key={chunk.chunk_id}
          className="flex flex-col gap-4 rounded-md border border-border p-4 lg:flex-row lg:items-stretch"
        >
          <div className="w-full shrink-0 space-y-3 lg:w-96">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">#{index + 1}</span>
              <span className="text-sm">{chunk.title}</span>
              {chunk.kb_name || chunk.kb_id ? (
                <Badge className="border border-border bg-background text-foreground">
                  {chunk.kb_name ?? chunk.kb_id.slice(0, 8)}
                </Badge>
              ) : null}
              {chunk.index_version ? <Badge>索引 {chunk.index_version}</Badge> : null}
            </div>
            <ChunkMetricsList metrics={buildChunkMetrics(chunk)} />
            <ChunkIds documentId={chunk.document_id} chunkId={chunk.chunk_id} />
            {chunk.metadata?.heading_path ? (
              <p className="text-xs text-muted-foreground">
                所属章节：
                <span className="text-foreground">{String(chunk.metadata.heading_path)}</span>
              </p>
            ) : null}
            {chunk.metadata?.chunk_type ? (
              <p className="text-xs text-muted-foreground">
                块类型：
                <span className="text-foreground">{String(chunk.metadata.chunk_type)}</span>
              </p>
            ) : null}
          </div>
          <div className="min-w-0 flex-1 space-y-3">
            <MarkdownContent content={chunk.content} />
            {chunk.context ? <ChunkContextDetails chunk={chunk} /> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

const contextRelationLabels = {
  anchor: "原始命中",
  previous: "前一片段",
  next: "后一片段",
  same_section: "同章节",
  parent_section: "父章节",
  reference: "明确引用",
} as const;

function ChunkContextDetails({ chunk }: { chunk: RetrievedChunkData }) {
  const context = chunk.context;
  if (!context) return null;

  return (
    <details className="rounded-md border border-border bg-muted/30 p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        上下文补充（{Math.max(context.sources.length - 1, 0)} 条）
      </summary>
      <div className="mt-3 space-y-3">
        <div className="space-y-2">
          {context.sources.map((source) => (
            <div key={`${source.relation}-${source.chunk_id}`} className="rounded border bg-background p-2">
              <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <Badge>{contextRelationLabels[source.relation]}</Badge>
                {source.title ? <span>{source.title}</span> : null}
                <span>Chunk {source.chunk_id}</span>
              </div>
              {source.content ? <MarkdownContent content={source.content} /> : null}
            </div>
          ))}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">组合后的上下文</div>
          <MarkdownContent content={context.content} />
        </div>
        <div className="space-y-1 text-xs text-muted-foreground">
          {chunk.order_index != null ? <p>片段顺序：{chunk.order_index}</p> : null}
          {chunk.parent_section_id ? <p>父章节 ID：{chunk.parent_section_id}</p> : null}
          {context.latency_ms != null ? <p>扩展耗时：{context.latency_ms} ms</p> : null}
          {context.degraded ? <p className="text-amber-700">上下文扩展已降级</p> : null}
          {context.error ? <p className="text-amber-700">{context.error}</p> : null}
        </div>
      </div>
    </details>
  );
}

function EvidencePackCard({
  pack,
  metadata,
}: {
  pack: EvidencePack | null;
  metadata: RetrieveData["metadata"]["evidence"];
}) {
  if (!metadata?.enabled && !pack) {
    return null;
  }
  const evidenceMetadata = metadata ?? {
    degraded: false,
    error: null,
    candidate_count: 0,
    evidence_count: pack?.items.length ?? 0,
    output_chars: pack?.items.reduce((total, item) => total + item.content.length, 0) ?? 0,
    latency_ms: 0,
    context_sources_included: false,
    budget_exceeded: false,
  };
  const itemsById = new Map(pack?.items.map((item) => [item.evidence_id, item]) ?? []);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>证据包</CardTitle>
        <Badge>{pack?.status ?? (evidenceMetadata.degraded ? "degraded" : "missing")}</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {evidenceMetadata.degraded ? (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            证据编排已降级，原检索结果不受影响
            {evidenceMetadata.error ? `：${evidenceMetadata.error}` : ""}
          </div>
        ) : null}
        {pack?.missing_aspects.length ? (
          <p className="text-sm text-amber-700">
            缺失要点：{pack.missing_aspects.join("、")}
          </p>
        ) : null}
        {pack?.groups.map((group) => (
          <div key={group.aspect} className="space-y-2 rounded-md border p-3">
            <div className="flex items-center gap-2 font-medium">
              <span>{group.aspect}</span>
              <Badge>{group.covered ? "已覆盖" : "未覆盖"}</Badge>
            </div>
            {group.evidence.map((reference) => {
              const item = itemsById.get(reference.evidence_id);
              if (!item) return null;
              return (
                <div
                  key={`${group.aspect}-${reference.evidence_id}`}
                  className="rounded border bg-muted/30 p-3"
                >
                  <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <Badge>{reference.evidence_id}</Badge>
                    <Badge>{reference.role}</Badge>
                    <Badge>
                      {item.retrieved_rank != null
                        ? `TopK #${item.retrieved_rank}`
                        : "TopK 外"}
                    </Badge>
                    <span>{item.title}</span>
                    {item.heading_path ? <span>{item.heading_path}</span> : null}
                    <span>{item.source}</span>
                  </div>
                  <MarkdownContent content={item.content} />
                </div>
              );
            })}
          </div>
        ))}
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
          <span>候选 {evidenceMetadata.candidate_count}</span>
          <span>证据 {evidenceMetadata.evidence_count}</span>
          <span>正文 {evidenceMetadata.output_chars} 字符</span>
          <span>耗时 {evidenceMetadata.latency_ms} ms</span>
          {evidenceMetadata.context_sources_included ? <span>已包含 context sources</span> : null}
          {evidenceMetadata.budget_exceeded ? (
            <span className="text-amber-700">达到输出预算</span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function RetrieveSummaryCard({ data }: { data: RetrieveData }) {
  const retrieval = data.metadata.retrieval;
  const rerank = data.metadata.rerank;
  const queryProcessing = data.metadata.query_processing;
  const tenantPolicy = data.metadata.tenant_policy;

  return (
    <Card>
      <CardHeader>
        <CardTitle>运行摘要</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <Badge>mode {retrieval?.mode ?? "—"}</Badge>
          <Badge>top_k {data.metadata.top_k}</Badge>
          {retrieval?.multi_kb ? <Badge>multi_kb ×{retrieval.kb_count ?? data.kb_ids.length}</Badge> : null}
          {retrieval?.fusion ? <Badge>fusion {retrieval.fusion}</Badge> : null}
          <Badge>服务端耗时 {data.metadata.latency_ms} ms</Badge>
          <Badge>{data.metadata.vector_store}</Badge>
        </div>

        {tenantPolicy ? (
          <div className="rounded-md border border-border bg-muted/40 p-3">
            <div className="mb-1 font-medium">租户策略（tenant_policy）</div>
            <div className="space-y-1 text-muted-foreground">
              <p className="flex flex-wrap items-center gap-2">
                <span>套餐</span>
                <PlanBadge plan={tenantPolicy.plan} showCode />
              </p>
              <p>
                本次 profile：
                <span className="text-foreground">{tenantPolicy.retrieve_profile}</span>
              </p>
              <p>
                实际 mode：<span className="text-foreground">{tenantPolicy.effective_mode}</span>
              </p>
              <p>
                rerank：
                <span className="text-foreground">
                  {tenantPolicy.effective_rerank ? "已启用" : "未启用"}
                  {tenantPolicy.rerank_disabled_reason
                    ? `（${tenantPolicy.rerank_disabled_reason}）`
                    : ""}
                </span>
              </p>
              <p>
                证据编排：
                <span className="text-foreground">
                  {tenantPolicy.effective_evidence ? "已启用" : "未启用"}
                  {tenantPolicy.evidence_disabled_reason
                    ? `（${tenantPolicy.evidence_disabled_reason}）`
                    : ""}
                </span>
              </p>
              <p>
                query 改写：
                <span className="text-foreground">
                  {tenantPolicy.effective_query_rewrite ? "已启用" : "未启用"}
                  {tenantPolicy.query_rewrite_disabled_reason
                    ? `（${tenantPolicy.query_rewrite_disabled_reason}）`
                    : ""}
                </span>
              </p>
              {tenantPolicy.max_kb_per_retrieve != null ? (
                <p>
                  联合检索库数：
                  <span className="text-foreground">
                    {tenantPolicy.actual_kb_count ?? data.kb_ids.length} /{" "}
                    {tenantPolicy.max_kb_per_retrieve}
                  </span>
                </p>
              ) : null}
            </div>
          </div>
        ) : null}

        {queryProcessing ? (
          <div className="rounded-md border border-border bg-muted/40 p-3">
            <div className="mb-1 font-medium">问句处理</div>
            <div className="space-y-1 text-muted-foreground">
              <p>
                原话：<span className="text-foreground">{queryProcessing.raw_query}</span>
              </p>
              {queryProcessing.effective_query !== queryProcessing.raw_query ? (
                <p>
                  改写检索句：
                  <span className="text-foreground">{queryProcessing.effective_query}</span>
                </p>
              ) : null}
              {queryProcessing.search_query !== queryProcessing.effective_query ? (
                <p>
                  最终检索句：
                  <span className="font-medium text-foreground">
                    {queryProcessing.search_query}
                  </span>
                </p>
              ) : (
                <p>
                  最终检索句：
                  <span className="text-foreground">{queryProcessing.search_query}</span>
                </p>
              )}
              {queryProcessing.synonym_applied ? (
                <p>
                  词表扩展：
                  <span className="text-foreground">
                    {queryProcessing.synonym_expansions.join("、")}
                  </span>
                </p>
              ) : null}
              {queryProcessing.enabled ? (
                <>
                  <p>改写耗时：{queryProcessing.latency_ms} ms</p>
                  {queryProcessing.degraded ? (
                    <p className="text-amber-700">
                      改写失败，已使用原话检索
                      {queryProcessing.degraded_reason
                        ? `：${queryProcessing.degraded_reason}`
                        : ""}
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
          </div>
        ) : null}

        {retrieval ? (
          <div className="rounded-md border border-border bg-muted/40 p-3">
            <div className="mb-1 font-medium">实际召回配置</div>
            <div className="flex flex-wrap gap-2 text-muted-foreground">
              {retrieval.vector_top_k != null ? <span>Vector Top{retrieval.vector_top_k}</span> : null}
              {retrieval.bm25_top_k != null ? <span>BM25 Top{retrieval.bm25_top_k}</span> : null}
              {retrieval.fusion ? (
                <span>{retrieval.fusion_scope === "global" ? "全局 " : ""}{retrieval.fusion} / rrf_k={retrieval.rrf_k}</span>
              ) : null}
              {retrieval.vector_count != null ? <span>vector: {retrieval.vector_count}</span> : null}
              {retrieval.bm25_count != null ? <span>bm25: {retrieval.bm25_count}</span> : null}
              {retrieval.fused_count != null ? <span>fused: {retrieval.fused_count}</span> : null}
              {retrieval.embedding_request_count != null ? (
                <span>Embedding调用: {retrieval.embedding_request_count}</span>
              ) : null}
              {retrieval.vector_request_count != null ? (
                <span>Vector调用: {retrieval.vector_request_count}</span>
              ) : null}
              {retrieval.bm25_request_count != null ? (
                <span>BM25调用: {retrieval.bm25_request_count}</span>
              ) : null}
              {retrieval.hybrid_latency_ms != null ? (
                <span>Hybrid墙钟耗时: {retrieval.hybrid_latency_ms} ms</span>
              ) : retrieval.cost_ms != null ? (
                <span>召回耗时: {retrieval.cost_ms} ms</span>
              ) : null}
            </div>
          </div>
        ) : null}

        {retrieval?.degraded ? (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">
            hybrid degraded
            {retrieval.degraded_reason ? `：${retrieval.degraded_reason}` : ""}
          </div>
        ) : null}

        <div className="rounded-md border border-border bg-muted/40 p-3">
          <div className="mb-1 font-medium">rerank</div>
          <div className="space-y-1 text-muted-foreground">
            <p>
              {rerank.enabled ? "已启用" : "未启用"} / {rerank.provider}
              {rerank.model ? ` / ${rerank.model}` : ""}
            </p>
            {rerank.top_n != null ? <p>top_n: {rerank.top_n}</p> : null}
            {rerank.candidate_count != null ? (
              <p>candidate_count: {rerank.candidate_count}</p>
            ) : null}
            {rerank.degraded ? (
              <p className={cn("text-destructive")}>
                rerank degraded{rerank.error ? `：${rerank.error}` : ""}
              </p>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
