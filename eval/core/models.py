from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tenant.retrieve_presets import RETRIEVE_PROFILE_PRESETS

CaseType = Literal[
    "direct_fact",
    "paraphrase",
    "multi_document",
    "independent_multi_aspect",
    "dependent_multi_hop",
    "control",
]
RetrieveProfile = Literal["speed", "balanced", "quality", "custom"]
EvalRerankProvider = Literal["noop", "llm", "qwen3"]
EvalExecutionMode = Literal["retrieve", "research_no_supplement", "research"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeBaseRef(StrictModel):
    kb_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)


class SuiteDefinition(StrictModel):
    description: str = ""
    case_ids: list[str] | Literal["all"]


class EvalAspect(StrictModel):
    aspect_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    acceptable_chunk_ids: list[str] = Field(default_factory=list)


class EvalDependency(StrictModel):
    discover_from_chunk_ids: list[str] = Field(default_factory=list)
    target_chunk_ids: list[str] = Field(default_factory=list)


class EvalCase(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    kb_alias: str | None = None
    kb_aliases: list[str] | None = None
    suite: str
    case_type: CaseType
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    expected_documents: list[str] = Field(default_factory=list)
    expected_headings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    required_aspects: list[EvalAspect] = Field(default_factory=list)
    expected_status: Literal["complete", "partial", "missing"] | None = None
    requires_multi_hop: bool = False
    dependency: EvalDependency | None = None
    relevance: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kb_scope(self) -> Self:
        aliases = self.resolved_kb_aliases()
        if not aliases:
            raise ValueError("kb_alias 或 kb_aliases 必须提供一个")
        if len(set(aliases)) != len(aliases):
            raise ValueError("kb_aliases 不能包含重复值")
        if self.kb_alias and self.kb_aliases:
            raise ValueError("kb_alias 和 kb_aliases 不能同时提供")
        return self

    def resolved_kb_aliases(self) -> list[str]:
        if self.kb_aliases:
            return [item.strip() for item in self.kb_aliases if item.strip()]
        if self.kb_alias and self.kb_alias.strip():
            return [self.kb_alias.strip()]
        return []


class EvalDataset(StrictModel):
    schema_version: str
    name: str
    version: str
    description: str = ""
    knowledge_bases: dict[str, KnowledgeBaseRef]
    suites: dict[str, SuiteDefinition]
    cases: list[EvalCase]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if not self.knowledge_bases:
            raise ValueError("knowledge_bases 不能为空")
        if not self.suites:
            raise ValueError("suites 不能为空")

        case_ids = [case.id for case in self.cases]
        duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(f"case ID 重复: {', '.join(duplicate_ids)}")

        known_case_ids = set(case_ids)
        known_aliases = set(self.knowledge_bases)
        for case in self.cases:
            unknown_aliases = set(case.resolved_kb_aliases()) - known_aliases
            if unknown_aliases:
                raise ValueError(
                    f"case {case.id} 使用未知知识库别名: {', '.join(sorted(unknown_aliases))}"
                )
            if case.suite not in self.suites:
                raise ValueError(f"case {case.id} 使用未知 suite: {case.suite}")

        for suite_name, suite in self.suites.items():
            if suite.case_ids == "all":
                continue
            unknown_ids = set(suite.case_ids) - known_case_ids
            if unknown_ids:
                raise ValueError(
                    f"suite {suite_name} 引用了不存在的 case: "
                    f"{', '.join(sorted(unknown_ids))}"
                )
            if len(suite.case_ids) != len(set(suite.case_ids)):
                raise ValueError(f"suite {suite_name} 包含重复 case ID")
        return self

    def cases_for_suite(self, suite_name: str) -> list[EvalCase]:
        suite = self.suites.get(suite_name)
        if suite is None:
            raise ValueError(f"未知 suite: {suite_name}")
        if suite.case_ids == "all":
            return list(self.cases)
        by_id = {case.id: case for case in self.cases}
        return [by_id[case_id] for case_id in suite.case_ids]


class ExperimentGroup(StrictModel):
    label: str
    index_version: str = Field(default="v1", pattern=r"^v[1-9][0-9]*$")
    profile: RetrieveProfile = "custom"
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_options: dict[str, Any] = Field(default_factory=dict)
    rerank_options: dict[str, Any] = Field(default_factory=dict)
    evidence_options: dict[str, Any] = Field(default_factory=dict)
    query_options: dict[str, Any] = Field(default_factory=dict)
    rerank_provider: EvalRerankProvider | None = None
    execution_mode: EvalExecutionMode = "retrieve"

    @model_validator(mode="before")
    @classmethod
    def resolve_named_profile(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        profile = value.get("profile", "custom")
        if profile not in {"speed", "balanced", "quality"}:
            return value
        advanced_fields = {
            "top_k",
            "retrieval_options",
            "rerank_options",
            "evidence_options",
            "query_options",
        }
        conflicts = sorted(advanced_fields & set(value))
        if conflicts:
            raise ValueError(
                "命名评测档位只声明 profile，不能复制高级参数: "
                + ", ".join(conflicts)
            )
        preset = RETRIEVE_PROFILE_PRESETS[profile]
        assert preset is not None
        return {
            **value,
            "top_k": preset.top_k,
            "retrieval_options": dict(preset.retrieval_options or {}),
            "rerank_options": dict(preset.rerank_options or {}),
            "evidence_options": dict(preset.evidence_options or {}),
            "query_options": dict(preset.query_options or {}),
        }

    def comparable_config(self) -> dict[str, Any]:
        return self.model_dump(exclude={"label"}, exclude_none=True)


class MetricGuardrail(StrictModel):
    min_delta: float


class CostLimits(StrictModel):
    max_p95_increase_ratio: float = Field(default=0.30, ge=0)
    max_model_calls_increase: float = Field(default=1.0, ge=0)


class DecisionRules(StrictModel):
    min_primary_improvement: float = Field(default=0.03, ge=0)
    quality_guardrails: dict[str, MetricGuardrail] = Field(default_factory=dict)
    cost_limits: CostLimits = Field(default_factory=CostLimits)
    max_failed_cases: int = Field(default=0, ge=0)


class ExperimentConfig(StrictModel):
    schema_version: str
    experiment_id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    target: str
    changed_fields: list[str] = Field(default_factory=list)
    dataset: str
    suite: str
    primary_metric: Literal["context_precision", "context_recall"]
    concurrency: int = Field(default=4, ge=1, le=20)
    ragas_concurrency: int = Field(default=4, ge=1, le=20)
    warmup_cases: int = Field(default=0, ge=0, le=10)
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    shared_candidate_top_k: int | None = Field(default=None, ge=1, le=50)
    baseline_run_dir: str | None = None
    baseline_group: Literal["baseline", "candidate"] = "baseline"
    baseline: ExperimentGroup
    candidate: ExperimentGroup
    decision: DecisionRules = Field(default_factory=DecisionRules)

    @model_validator(mode="after")
    def validate_group_differences(self) -> Self:
        if not self.changed_fields and self.baseline_run_dir is None:
            raise ValueError("非冻结基线实验必须声明 changed_fields")
        actual = diff_paths(
            self.baseline.comparable_config(),
            self.candidate.comparable_config(),
        )
        allowed = set(self.changed_fields)
        undeclared = actual - allowed
        unchanged = allowed - actual
        if undeclared:
            raise ValueError(
                "baseline/candidate 存在未声明差异: " + ", ".join(sorted(undeclared))
            )
        if unchanged:
            raise ValueError(
                "changed_fields 中的字段实际没有变化: " + ", ".join(sorted(unchanged))
            )

        if self.baseline_run_dir is not None and actual:
            raise ValueError("冻结基线实验的 baseline/candidate 配置必须完全相同")

        providers = (
            self.baseline.rerank_provider,
            self.candidate.rerank_provider,
        )
        if self.target == "evidence_orchestration":
            self._validate_evidence_orchestration()
            return self
        if self.target == "rerank_effect_v2":
            self._validate_rerank_effect_v2()
            return self
        if self.target in {"research_effect_v2", "research_supplement_v2"}:
            self._validate_research_experiment()
            return self
        if any(provider is not None for provider in providers):
            if not all(provider is not None for provider in providers):
                raise ValueError("精排对比实验的 baseline 和 candidate 都必须声明 rerank_provider")
            if self.shared_candidate_top_k is None:
                raise ValueError("精排对比实验必须声明 shared_candidate_top_k")
            if self.warmup_cases != 0:
                raise ValueError("精排对比实验每道题每组只运行一次，warmup_cases 必须为0")
            if self.shared_candidate_top_k < max(
                self.baseline.top_k,
                self.candidate.top_k,
            ):
                raise ValueError("shared_candidate_top_k 不能小于最终 top_k")
            for name, group in (
                ("baseline", self.baseline),
                ("candidate", self.candidate),
            ):
                if group.retrieval_options.get("mode") != "hybrid":
                    raise ValueError(f"{name} 精排实验必须使用 hybrid 检索")
                if (
                    group.query_options.get("enabled") is not False
                    and self.target != "opt20_prechange_quality_baseline"
                ):
                    raise ValueError(f"{name} 精排实验必须关闭 Query Rewrite")
                if group.query_options.get("synonym_enabled") is not False:
                    raise ValueError(f"{name} 精排实验必须关闭同义词扩展")
                if group.evidence_options.get("enabled") is not False:
                    raise ValueError(f"{name} 精排实验必须关闭证据编排")
                if group.rerank_options.get("enabled") is not True:
                    raise ValueError(f"{name} 精排实验必须启用重排")
                if group.rerank_options.get("top_n") != group.top_k:
                    raise ValueError(f"{name} 的 rerank top_n 必须等于最终 top_k")
        elif self.shared_candidate_top_k is not None:
            raise ValueError("只有精排对比实验可以声明 shared_candidate_top_k")
        return self

    def _validate_research_experiment(self) -> None:
        if self.shared_candidate_top_k is not None:
            raise ValueError("Research实验不使用跨组共同RRF候选字段")
        if self.warmup_cases != 0:
            raise ValueError("Research实验不执行warm-up")
        expected_modes = (
            ("retrieve", "research")
            if self.target == "research_effect_v2"
            else ("research_no_supplement", "research")
        )
        actual_modes = (self.baseline.execution_mode, self.candidate.execution_mode)
        if actual_modes != expected_modes:
            raise ValueError(
                "Research实验执行模式错误: "
                f"expected={expected_modes}, actual={actual_modes}"
            )
        for name, group in (("baseline", self.baseline), ("candidate", self.candidate)):
            if group.index_version != "v2":
                raise ValueError(f"{name} Research实验必须使用v2索引")
            if group.top_k != 10:
                raise ValueError(f"{name} Research实验最终上限必须为Top10")
            if group.retrieval_options.get("mode") != "hybrid":
                raise ValueError(f"{name} Research实验必须使用hybrid检索")
            if group.query_options.get("enabled") is not False:
                raise ValueError(f"{name} Research实验必须关闭Query Rewrite")
            if group.query_options.get("synonym_enabled") is not False:
                raise ValueError(f"{name} Research实验必须关闭同义词扩展")
            if group.rerank_provider != "qwen3":
                raise ValueError(f"{name} Research实验必须使用Qwen3精排")
            if group.rerank_options.get("enabled") is not True:
                raise ValueError(f"{name} Research实验必须启用精排")
            if group.rerank_options.get("top_n") != 10:
                raise ValueError(f"{name} Research实验精排必须返回Top10")
            if group.evidence_options.get("enabled") is not True:
                raise ValueError(f"{name} Research实验必须输出EvidencePack")
            if group.evidence_options.get("max_items") != 10:
                raise ValueError(f"{name} Research实验EvidencePack上限必须为10")

    def _validate_evidence_orchestration(self) -> None:
        if self.shared_candidate_top_k is None:
            raise ValueError("证据编排实验必须声明 shared_candidate_top_k")
        if self.warmup_cases != 0:
            raise ValueError("证据编排实验每道题只运行一次，warmup_cases 必须为0")
        if self.shared_candidate_top_k < max(self.baseline.top_k, self.candidate.top_k):
            raise ValueError("shared_candidate_top_k 不能小于最终 top_k")
        if self.baseline.rerank_provider != "qwen3" or self.candidate.rerank_provider != "qwen3":
            raise ValueError("证据编排实验的两组都必须使用 qwen3 精排")
        for name, group in (("baseline", self.baseline), ("candidate", self.candidate)):
            if group.index_version != "v2":
                raise ValueError(f"{name} 证据编排实验必须使用 v2 索引")
            if group.retrieval_options.get("mode") != "hybrid":
                raise ValueError(f"{name} 证据编排实验必须使用 hybrid 检索")
            if group.query_options.get("enabled") is not False:
                raise ValueError(f"{name} 证据编排实验必须关闭 Query Rewrite")
            if group.query_options.get("synonym_enabled") is not False:
                raise ValueError(f"{name} 证据编排实验必须关闭同义词扩展")
            if group.rerank_options.get("enabled") is not True:
                raise ValueError(f"{name} 证据编排实验必须启用精排")
            if group.rerank_options.get("top_n") != group.top_k:
                raise ValueError(f"{name} 的 rerank top_n 必须等于最终 top_k")
        if self.baseline.evidence_options.get("enabled") is not False:
            raise ValueError("baseline 必须关闭证据编排")
        if self.candidate.evidence_options.get("enabled") is not True:
            raise ValueError("candidate 必须开启证据编排")

    def _validate_rerank_effect_v2(self) -> None:
        if self.shared_candidate_top_k != 20:
            raise ValueError("精排开关实验必须使用共同候选 Top20")
        if self.warmup_cases != 0:
            raise ValueError("精排开关实验不执行 warm-up")
        if self.baseline.rerank_provider != "noop":
            raise ValueError("baseline 必须使用 RRF 原始顺序")
        if self.candidate.rerank_provider != "qwen3":
            raise ValueError("candidate 必须使用 Qwen3 精排")
        for name, group in (("baseline", self.baseline), ("candidate", self.candidate)):
            if group.index_version != "v2":
                raise ValueError(f"{name} 必须使用 v2 索引")
            if group.top_k != 10:
                raise ValueError(f"{name} 最终必须返回 Top10")
            if group.retrieval_options.get("mode") != "hybrid":
                raise ValueError(f"{name} 必须使用 hybrid 检索")
            if group.query_options.get("enabled") is not False:
                raise ValueError(f"{name} 必须关闭 Query Rewrite")
            if group.query_options.get("synonym_enabled") is not False:
                raise ValueError(f"{name} 必须关闭同义词扩展")
            if group.evidence_options.get("enabled") is not False:
                raise ValueError(f"{name} 必须关闭证据编排")
        if self.baseline.rerank_options.get("enabled") is not False:
            raise ValueError("baseline 必须关闭精排")
        if self.candidate.rerank_options.get("enabled") is not True:
            raise ValueError("candidate 必须开启精排")
        if self.candidate.rerank_options.get("top_n") != 10:
            raise ValueError("candidate 精排必须返回 Top10")


def diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.add(path)
                continue
            differences.update(diff_paths(left[key], right[key], path))
        return differences
    if left != right:
        return {prefix}
    return set()
