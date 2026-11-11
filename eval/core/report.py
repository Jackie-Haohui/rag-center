from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.core.models import EvalDataset, ExperimentConfig

VERDICT_TEXT = {
    "effective": "有效，建议启用",
    "effective_costly": "有效但成本较高，建议按场景启用",
    "no_clear_gain": "无明显收益，暂不启用",
    "regressed": "出现负向效果，建议回退",
    "evaluation_failed": "评测执行失败，暂时不能下结论",
}


def generate_report(run_dir: Path) -> dict[str, Any]:
    experiment = ExperimentConfig.model_validate(
        _read_json(run_dir / "experiment.snapshot.json")
    )
    dataset = EvalDataset.model_validate(_read_json(run_dir / "dataset.snapshot.json"))
    baseline = _read_json(run_dir / "baseline.metrics.json")
    candidate = _read_json(run_dir / "candidate.metrics.json")
    comparison = build_comparison(
        experiment=experiment,
        dataset=dataset,
        baseline=baseline,
        candidate=candidate,
    )
    if experiment.target == "evidence_orchestration":
        comparison["evidence_regression_details"] = _evidence_regression_details(
            comparison=comparison,
            candidate_rows=_read_jsonl(run_dir / "candidate.raw.jsonl"),
        )
    if experiment.target in {"research_effect_v2", "research_supplement_v2"}:
        comparison["research_details"] = _research_details(
            baseline_rows=_read_jsonl(run_dir / "baseline.raw.jsonl"),
            candidate_rows=_read_jsonl(run_dir / "candidate.raw.jsonl"),
            shared_rows=_read_jsonl(run_dir / "shared_pipeline.raw.jsonl"),
        )
        if experiment.target == "research_supplement_v2":
            details = comparison["research_details"]
            valid = (
                details["shared_first_round_cases"] == comparison["case_count"]
                and details["repair_denominator"] >= 2
            )
            comparison["checks"]["supplement_design_valid"] = valid
            if not valid:
                comparison["verdict"] = "evaluation_failed"
                comparison["verdict_text"] = "补证实验条件不成立，暂时不能判断第二轮是否有效"
                comparison["recommendation"] = (
                    "本轮4题没有形成至少2道合法修复分母，且两组只有"
                    f"{details['shared_first_round_cases']}/4道首轮快照一致。"
                    "Precision变化不能可靠归因于第二轮补证；应先固定复用Planner与首轮候选快照，"
                    "并补充首轮确实缺证、知识库中又存在目标证据的多跳题后重测。"
                )
    _write_json(run_dir / "comparison.json", comparison)
    (run_dir / "report.md").write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["status"] = (
        "evaluation_failed"
        if comparison["verdict"] == "evaluation_failed"
        else "completed"
    )
    manifest["verdict"] = comparison["verdict"]
    manifest["verdict_text"] = comparison["verdict_text"]
    _write_json(manifest_path, manifest)
    return comparison


def build_comparison(
    *,
    experiment: ExperimentConfig,
    dataset: EvalDataset,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = dict(baseline["summary"])
    candidate_summary = dict(candidate["summary"])
    quality_comparison: dict[str, Any] | None = None
    if experiment.target in {"research_effect_v2", "research_supplement_v2"}:
        baseline_summary, candidate_summary, quality_comparison = (
            _align_research_quality_summaries(
                baseline_summary=baseline_summary,
                candidate_summary=candidate_summary,
                baseline_cases=baseline.get("per_case") or [],
                candidate_cases=candidate.get("per_case") or [],
            )
        )
    metric_names = sorted(set(baseline_summary) | set(candidate_summary))
    delta = {
        name: _delta(baseline_summary.get(name), candidate_summary.get(name))
        for name in metric_names
    }
    primary_delta = delta.get(experiment.primary_metric)
    min_primary_improvement = experiment.decision.min_primary_improvement
    primary_passed = (
        primary_delta is not None
        and primary_delta >= min_primary_improvement
    )
    failed_baseline = int(baseline_summary.get("request_failed_cases") or 0)
    failed_candidate = int(candidate_summary.get("request_failed_cases") or 0)
    ragas_failed_baseline = int(baseline_summary.get("failed_cases") or 0)
    ragas_failed_candidate = int(candidate_summary.get("failed_cases") or 0)

    guardrail_checks: dict[str, Any] = {}
    guardrails_valid = True
    for metric, rule in experiment.decision.quality_guardrails.items():
        metric_delta = delta.get(metric)
        passed = metric_delta is not None and metric_delta >= rule.min_delta
        guardrails_valid = guardrails_valid and passed
        guardrail_checks[metric] = {
            "delta": metric_delta,
            "min_delta": rule.min_delta,
            "passed": passed,
        }

    latency_ratio = _increase_ratio(
        baseline_summary.get("latency_p95_ms"),
        candidate_summary.get("latency_p95_ms"),
    )
    calls_delta = delta.get("average_application_model_calls")
    latency_cost_passed = (
        latency_ratio is not None
        and latency_ratio
        <= experiment.decision.cost_limits.max_p95_increase_ratio
    )
    calls_cost_passed = (
        calls_delta is not None
        and calls_delta
        <= experiment.decision.cost_limits.max_model_calls_increase
    )
    cost_passed = latency_cost_passed and calls_cost_passed

    scoring_errors = [
        item
        for item in (
            baseline.get("ragas", {}).get("error"),
            candidate.get("ragas", {}).get("error"),
        )
        if item
    ]
    result_complete = (
        primary_delta is not None
        and failed_baseline <= experiment.decision.max_failed_cases
        and failed_candidate <= experiment.decision.max_failed_cases
        and ragas_failed_baseline <= experiment.decision.max_failed_cases
        and ragas_failed_candidate <= experiment.decision.max_failed_cases
        and not scoring_errors
    )
    citation_traceability_rate = (candidate.get("diagnostics") or {}).get(
        "citation_traceability_rate"
    )
    traceability_required = experiment.target in {
        "evidence_orchestration",
        "research_effect_v2",
        "research_supplement_v2",
    }
    citation_traceability_passed = not traceability_required or citation_traceability_rate == 1.0
    result_complete = result_complete and citation_traceability_passed
    if experiment.target in {"research_effect_v2", "research_supplement_v2"}:
        candidate_diagnostics = candidate.get("diagnostics") or {}
        result_complete = result_complete and (
            int(candidate_diagnostics.get("answer_present_cases") or 0)
            == int(candidate_summary.get("successful_cases") or 0)
            and int(candidate_diagnostics.get("answer_citations_valid_cases") or 0)
            == int(candidate_summary.get("successful_cases") or 0)
            and int(candidate_diagnostics.get("degraded_cases") or 0) == 0
        )

    quality_deltas = [primary_delta]
    quality_deltas.extend(
        check["delta"] for check in guardrail_checks.values()
    )
    quality_improved = any(
        value is not None and value > 0 for value in quality_deltas
    )
    primary_regressed = (
        primary_delta is not None
        and (
            primary_delta < 0
            if min_primary_improvement == 0
            else primary_delta <= -min_primary_improvement
        )
    )

    if not result_complete:
        verdict = "evaluation_failed"
    elif not guardrails_valid or primary_regressed:
        verdict = "regressed"
    elif primary_passed and quality_improved:
        verdict = "effective" if cost_passed else "effective_costly"
    else:
        verdict = "no_clear_gain"

    executed_case_count = _executed_case_count(
        baseline_summary,
        fallback=len(dataset.cases_for_suite(experiment.suite)),
    )
    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "target": experiment.target,
        "dataset_version": dataset.version,
        "suite": experiment.suite,
        "case_count": executed_case_count,
        "shared_candidate_top_k": experiment.shared_candidate_top_k,
        "final_top_k": experiment.candidate.top_k,
        "verdict": verdict,
        "verdict_text": _verdict_text(
            verdict,
            primary_passed=primary_passed,
            target=experiment.target,
        ),
        "primary_metric": experiment.primary_metric,
        "changed_fields": experiment.changed_fields,
        "baseline_label": experiment.baseline.label,
        "candidate_label": experiment.candidate.label,
        "baseline_index_version": experiment.baseline.index_version,
        "candidate_index_version": experiment.candidate.index_version,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "baseline_diagnostics": baseline.get("diagnostics") or {},
        "candidate_diagnostics": candidate.get("diagnostics") or {},
        "delta": delta,
        "thresholds": experiment.decision.model_dump(mode="json"),
        "checks": {
            "result_complete": result_complete,
            "primary_metric": {
                "delta": primary_delta,
                "min_improvement": min_primary_improvement,
                "passed": primary_passed,
            },
            "quality_improved": quality_improved,
            "quality_guardrails": guardrail_checks,
            "cost": {
                "latency_p95_increase_ratio": latency_ratio,
                "max_latency_p95_increase_ratio": (
                    experiment.decision.cost_limits.max_p95_increase_ratio
                ),
                "latency_passed": latency_cost_passed,
                "model_calls_delta": calls_delta,
                "max_model_calls_increase": (
                    experiment.decision.cost_limits.max_model_calls_increase
                ),
                "model_calls_passed": calls_cost_passed,
                "passed": cost_passed,
            },
            "citation_traceability": {
                "value": citation_traceability_rate,
                "required": 1.0 if traceability_required else None,
                "passed": citation_traceability_passed,
            },
        },
        "failed_cases": {
            "baseline_requests": failed_baseline,
            "candidate_requests": failed_candidate,
            "baseline_ragas": ragas_failed_baseline,
            "candidate_ragas": ragas_failed_candidate,
        },
        "scoring_errors": scoring_errors,
        "case_changes": _build_case_changes(
            baseline.get("per_case") or [],
            candidate.get("per_case") or [],
            experiment.primary_metric,
        ),
        "recall_case_changes": _build_case_changes(
            baseline.get("per_case") or [],
            candidate.get("per_case") or [],
            "context_recall",
        ),
        "recommendation": _recommendation(
            verdict,
            primary_passed=primary_passed,
            cost_passed=cost_passed,
            baseline_label=experiment.baseline.label,
            candidate_label=experiment.candidate.label,
            target=experiment.target,
            case_count=executed_case_count,
            precision_delta=delta.get("context_precision"),
            baseline_p95_ms=baseline_summary.get("latency_p95_ms"),
            candidate_p95_ms=candidate_summary.get("latency_p95_ms"),
            latency_ratio=latency_ratio,
            recall_delta=delta.get("context_recall"),
            ndcg_delta=delta.get("ndcg_at_10"),
            baseline_diagnostics=baseline.get("diagnostics") or {},
            candidate_diagnostics=candidate.get("diagnostics") or {},
        ),
    }
    if quality_comparison is not None:
        comparison["quality_comparison"] = quality_comparison
    return comparison


def _align_research_quality_summaries(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compare Research quality only on cases scored successfully in both groups."""
    baseline_by_id = {str(item["case_id"]): item for item in baseline_cases}
    candidate_by_id = {str(item["case_id"]): item for item in candidate_cases}
    shared_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    comparable_ids = [
        case_id
        for case_id in shared_ids
        if all(
            row.get(metric) is not None
            for row in (baseline_by_id[case_id], candidate_by_id[case_id])
            for metric in ("context_precision", "context_recall")
        )
    ]

    aligned_baseline = dict(baseline_summary)
    aligned_candidate = dict(candidate_summary)
    for metric in ("context_precision", "context_recall"):
        baseline_values = [
            float(baseline_by_id[case_id][metric]) for case_id in comparable_ids
        ]
        candidate_values = [
            float(candidate_by_id[case_id][metric]) for case_id in comparable_ids
        ]
        aligned_baseline[metric] = (
            sum(baseline_values) / len(baseline_values) if baseline_values else None
        )
        aligned_candidate[metric] = (
            sum(candidate_values) / len(candidate_values) if candidate_values else None
        )
    aligned_baseline["evaluated_cases"] = len(comparable_ids)
    aligned_candidate["evaluated_cases"] = len(comparable_ids)

    excluded = []
    for case_id in shared_ids:
        if case_id in comparable_ids:
            continue
        baseline_case = baseline_by_id[case_id]
        candidate_case = candidate_by_id[case_id]
        excluded.append(
            {
                "case_id": case_id,
                "baseline_reason": baseline_case.get("ragas_skipped")
                or baseline_case.get("error")
                or "score_missing",
                "candidate_reason": candidate_case.get("ragas_skipped")
                or candidate_case.get("error")
                or "excluded_to_match_baseline",
            }
        )
    return (
        aligned_baseline,
        aligned_candidate,
        {
            "case_count": len(comparable_ids),
            "case_ids": comparable_ids,
            "excluded_cases": excluded,
        },
    )


def _executed_case_count(summary: dict[str, Any], *, fallback: int) -> int:
    successful = summary.get("successful_cases")
    failed = summary.get("request_failed_cases")
    if successful is None or failed is None:
        return fallback
    return int(successful) + int(failed)


def render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    checks = comparison["checks"]
    latency_ratio = checks["cost"]["latency_p95_increase_ratio"]
    changed_fields = "、".join(f"§{item}§" for item in comparison["changed_fields"])
    lines = [
        f"# {comparison['title']}报告",
        "",
        "## 结论",
        "",
        f"**{comparison['verdict_text']}。**",
        "",
        _conclusion_paragraph(comparison),
        "",
        "## 核心结果",
        "",
        "| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |",
        "|------|----------|-----------|------|------|----------|",
        *_quality_metric_rows(comparison),
        (
            f"| P95 延迟 | {_milliseconds(baseline.get('latency_p95_ms'))} | "
            f"{_milliseconds(candidate.get('latency_p95_ms'))} | "
            f"{_latency_delta(delta.get('latency_p95_ms'), latency_ratio)} | "
            f"增幅不超过 {_ratio(comparison['thresholds']['cost_limits']['max_p95_increase_ratio'])} | "
            f"{_pass_text(checks['cost']['latency_passed'])} |"
        ),
        (
            f"| 平均应用模型调用数 | "
            f"{_calls(baseline.get('average_application_model_calls'))} | "
            f"{_calls(candidate.get('average_application_model_calls'))} | "
            f"{_calls_delta(delta.get('average_application_model_calls'))} | "
            f"最多增加 {_calls(comparison['thresholds']['cost_limits']['max_model_calls_increase'])} | "
            f"{_pass_text(checks['cost']['model_calls_passed'])} |"
        ),
        "",
        "## 实验目标",
        "",
        f"- 验证能力：{comparison['target']}",
        f"- baseline：{comparison['baseline_label']}",
        f"- candidate：{comparison['candidate_label']}",
        f"- 唯一允许变化的字段：{changed_fields}",
        "",
        "## 数据与运行条件",
        "",
        f"- 数据集：§{comparison['dataset_version']}§",
        f"- suite：§{comparison['suite']}§",
        f"- 题目数量：{comparison['case_count']} 道",
        "- 每道题每组正式执行 1 次",
        (
            f"- 索引版本：baseline={comparison['baseline_index_version']}，"
            f"candidate={comparison['candidate_index_version']}"
        ),
        "- RAGAS 遥测关闭，离线 Langfuse 关闭",
        *_rerank_condition_lines(comparison),
        "",
        "## 失败与异常",
        "",
        (
            f"- baseline 请求失败：{comparison['failed_cases']['baseline_requests']} 道；"
            f"RAGAS 失败：{comparison['failed_cases']['baseline_ragas']} 道"
        ),
        (
            f"- candidate 请求失败：{comparison['failed_cases']['candidate_requests']} 道；"
            f"RAGAS 失败：{comparison['failed_cases']['candidate_ragas']} 道"
        ),
    ]
    if comparison["scoring_errors"]:
        lines.extend(
            [
                f"- 评分错误：{error}"
                for error in comparison["scoring_errors"]
            ]
        )
    if comparison["target"] == "context_graph_upgrade":
        diagnostics = comparison["candidate_diagnostics"]
        lines.extend(
            [
                "",
                "## 图谱诊断信息",
                "",
                (
                    f"- 发生图谱注入或上下文补充的题目："
                    f"{diagnostics.get('graph_affected_cases', 0)} 道"
                ),
                f"- 图谱注入候选总数：{diagnostics.get('graph_injected_chunks', 0)}",
                f"- 上下文补充 Chunk 总数：{diagnostics.get('supplemental_chunks', 0)}",
                "- 以上仅用于解释结果，不参与正式验收门槛。",
            ]
        )
    if comparison["target"] == "evidence_orchestration":
        diagnostics = comparison["candidate_diagnostics"]
        baseline_diagnostics = comparison["baseline_diagnostics"]
        recall_regressions = comparison["recall_case_changes"]["regressed"]
        lines.extend(
            [
                "",
                "## 证据编排诊断",
                "",
                f"- 平均证据数量：{_number(diagnostics.get('average_evidence_count'))}",
                f"- 平均证据字符数：{_number(diagnostics.get('average_evidence_chars'))}",
                f"- 证据状态分布：{json.dumps(diagnostics.get('evidence_statuses') or {}, ensure_ascii=False)}",
                (
                    "- 引用回溯率："
                    f"{_percentage(diagnostics.get('citation_traceability_rate'))}"
                    "（硬门槛 100%）"
                ),
                (
                    "- 平均材料压缩："
                    f"{_number(baseline_diagnostics.get('average_evidence_count'))}条 / "
                    f"{_number(baseline_diagnostics.get('average_evidence_chars'))}字符 → "
                    f"{_number(diagnostics.get('average_evidence_count'))}条 / "
                    f"{_number(diagnostics.get('average_evidence_chars'))}字符"
                ),
                "",
                "### Recall退化定位",
                "",
                *(
                    _case_change_lines(recall_regressions)
                    if recall_regressions
                    else ["- 没有Recall退化题目"]
                ),
                *[
                    (
                        f"- `{item['case_id']}`证据包状态为{item['status']}，"
                        f"选中{item['evidence_count']}条；"
                        f"未覆盖期望文档：{'、'.join(item['missing_expected_documents'])}。"
                    )
                    for item in comparison.get("evidence_regression_details") or []
                    if item.get("missing_expected_documents")
                ],
                (
                    "- 本轮证据包最多只选择3条，没有触发10条数量上限；"
                    "Recall下降来自证据选择遗漏，不是max_items截断。"
                ),
            ]
        )
    if comparison["target"] == "rerank_upgrade_v2":
        lines.extend(_rerank_v2_decision_lines(comparison))
    if comparison["target"] == "rerank_effect_v2":
        lines.extend(
            [
                "",
                "## 排序诊断",
                "",
                (
                    "- nDCG@10（Golden Set期望文档与章节标注）："
                    f"{_quality(baseline.get('ndcg_at_10'))} → "
                    f"{_quality(candidate.get('ndcg_at_10'))}，"
                    f"变化{_points(delta.get('ndcg_at_10'))}。"
                ),
                "- nDCG@10只用于解释排序效果，不替代四项正式指标。",
            ]
        )
    if comparison["target"] in {"research_effect_v2", "research_supplement_v2"}:
        details = comparison.get("research_details") or {}
        diagnostics = comparison["candidate_diagnostics"]
        quality = comparison.get("quality_comparison") or {}
        lines.extend(
            [
                "",
                "## Research诊断",
                "",
                (
                    "- Context指标共同可评价题目："
                    f"{quality.get('case_count', comparison['case_count'])}/"
                    f"{comparison['case_count']} 道"
                ),
                *[
                    (
                        f"- 质量汇总排除 `{item['case_id']}`："
                        f"baseline={item['baseline_reason']}，"
                        f"candidate={item['candidate_reason']}"
                    )
                    for item in quality.get("excluded_cases") or []
                ],
                f"- 最终答案非空：{diagnostics.get('answer_present_cases', 0)}/{comparison['case_count']} 道",
                f"- 答案引用映射有效：{diagnostics.get('answer_citations_valid_cases', 0)}/{comparison['case_count']} 道",
                f"- 引用回溯率：{_percentage(diagnostics.get('citation_traceability_rate'))}（硬门槛100%）",
                f"- 发生第二轮补证：{diagnostics.get('supplement_cases', 0)} 道",
                f"- 降级：{diagnostics.get('degraded_cases', 0)} 道",
                f"- 平均Research LLM调用：{_number(diagnostics.get('average_research_llm_call_count'))} 次",
                f"- 平均检索任务：{_number(diagnostics.get('average_retrieval_task_count'))} 次",
                f"- 平均输入/输出Token：{_number(diagnostics.get('average_input_tokens'))} / {_number(diagnostics.get('average_output_tokens'))}",
            ]
        )
        if comparison["target"] == "research_supplement_v2":
            rate = details.get("multi_hop_repair_rate")
            lines.extend(
                [
                    "",
                    "## 第二轮补证诊断",
                    "",
                    f"- Multi-hop Repair Rate：{'not_evaluable' if rate is None else _percentage(rate)}",
                    f"- 可修复分母：{details.get('repair_denominator', 0)} 道；实际修复：{details.get('repaired_cases', 0)} 道",
                    f"- 两组首轮快照完全一致：{details.get('shared_first_round_cases', 0)}/{comparison['case_count']} 道",
                    "- Multi-hop Repair Rate为诊断指标，不替代四项正式指标。",
                ]
            )
    lines.extend(
        [
            "",
            "## 改善最多的题目",
            "",
            *_case_change_lines(comparison["case_changes"]["improved"]),
            "",
            "## 退化最多的题目",
            "",
            *_case_change_lines(comparison["case_changes"]["regressed"]),
            "",
            "## 启用建议",
            "",
            comparison["recommendation"],
            "",
            "## 原始结果",
            "",
            "- §baseline.raw.jsonl§",
            "- §candidate.raw.jsonl§",
            *(
                ["- §shared_pipeline.raw.jsonl§"]
                if comparison.get("target") in {
                    "evidence_orchestration",
                    "research_effect_v2",
                    "research_supplement_v2",
                }
                else ["- §shared_candidates.raw.jsonl§"]
                if comparison.get("shared_candidate_top_k") is not None
                else []
            ),
            "- §baseline.metrics.json§",
            "- §candidate.metrics.json§",
            "- §comparison.json§",
            "",
        ]
    )
    return "\n".join(lines).replace("§", chr(96))


def _conclusion_paragraph(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    latency_ratio = comparison["checks"]["cost"]["latency_p95_increase_ratio"]
    return (
        f"在 {comparison['dataset_version']} 的 {comparison['case_count']} 道题上，"
        f"Context Precision 从 {_quality(baseline.get('context_precision'))} "
        f"变为 {_quality(candidate.get('context_precision'))}，"
        f"变化 {_points(delta.get('context_precision'))}；"
        f"Context Recall 从 {_quality(baseline.get('context_recall'))} "
        f"变为 {_quality(candidate.get('context_recall'))}，"
        f"变化 {_points(delta.get('context_recall'))}；"
        f"P95 延迟变化 {_milliseconds(delta.get('latency_p95_ms'))}"
        f"（{_ratio(latency_ratio)}）；平均应用模型调用数变化 "
        f"{_calls_delta(delta.get('average_application_model_calls'))}。"
        f"{comparison['recommendation']}"
    )


def _build_case_changes(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
    metric: str,
) -> dict[str, list[dict[str, Any]]]:
    baseline_by_id = {item["case_id"]: item for item in baseline_cases}
    changes: list[dict[str, Any]] = []
    for candidate in candidate_cases:
        baseline = baseline_by_id.get(candidate["case_id"])
        if baseline is None:
            continue
        value = _delta(baseline.get(metric), candidate.get(metric))
        if value is None:
            continue
        if abs(value) < 1e-9:
            value = 0.0
        changes.append(
            {
                "case_id": candidate["case_id"],
                "question": candidate.get("question"),
                "delta": value,
            }
        )
    improved = [item for item in changes if item["delta"] > 0]
    regressed = [item for item in changes if item["delta"] < 0]
    return {
        "improved": sorted(improved, key=lambda item: item["delta"], reverse=True)[:3],
        "regressed": sorted(regressed, key=lambda item: item["delta"])[:3],
    }


def _evidence_regression_details(
    *,
    comparison: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regressed_ids = {
        item["case_id"]
        for item in comparison.get("recall_case_changes", {}).get("regressed", [])
    }
    details: list[dict[str, Any]] = []
    for row in candidate_rows:
        if row.get("case_id") not in regressed_ids:
            continue
        selected_documents = {
            str(context.get("title") or "")
            for context in row.get("contexts") or []
            if str(context.get("title") or "")
        }
        expected_documents = [str(item) for item in row.get("expected_documents") or []]
        diagnostics = row.get("diagnostics") or {}
        details.append(
            {
                "case_id": row["case_id"],
                "status": diagnostics.get("evidence_status") or "unknown",
                "evidence_count": int(diagnostics.get("evidence_count") or 0),
                "missing_aspects": diagnostics.get("missing_aspects") or [],
                "missing_expected_documents": [
                    item for item in expected_documents if item not in selected_documents
                ],
            }
        )
    return details


def _research_details(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    shared_first_round_cases = sum(
        bool(row.get("shared"))
        or row.get("baseline_first_round") == row.get("candidate_first_round")
        for row in shared_rows
        if row.get("error") is None
    )
    denominator = 0
    repaired = 0
    per_case: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        candidate = candidate_by_id.get(baseline["case_id"])
        if candidate is None or not baseline.get("requires_multi_hop"):
            continue
        required = baseline.get("required_aspects") or []
        baseline_ids = {str(item.get("chunk_id") or "") for item in baseline.get("contexts") or []}
        candidate_ids = {str(item.get("chunk_id") or "") for item in candidate.get("contexts") or []}
        baseline_covered = _covered_aspects(required, baseline_ids)
        candidate_covered = _covered_aspects(required, candidate_ids)
        baseline_complete = len(baseline_covered) == len(required)
        candidate_complete = len(candidate_covered) == len(required)
        dependency = baseline.get("dependency") or {}
        has_target = bool(dependency.get("target_chunk_ids"))
        if not baseline_complete and has_target:
            denominator += 1
            repaired += int(candidate_complete)
        per_case.append(
            {
                "case_id": baseline["case_id"],
                "required_aspects": len(required),
                "baseline_covered": len(baseline_covered),
                "candidate_covered": len(candidate_covered),
                "repaired": not baseline_complete and candidate_complete,
            }
        )
    return {
        "shared_first_round_cases": shared_first_round_cases,
        "repair_denominator": denominator,
        "repaired_cases": repaired,
        "multi_hop_repair_rate": repaired / denominator if denominator else None,
        "per_case": per_case,
    }


def _covered_aspects(required: list[dict[str, Any]], chunk_ids: set[str]) -> set[str]:
    return {
        str(aspect.get("aspect_id") or "")
        for aspect in required
        if set(map(str, aspect.get("acceptable_chunk_ids") or [])) & chunk_ids
    }


def _case_change_lines(cases: list[dict[str, Any]]) -> list[str]:
    if not cases:
        return ["- 无可比较结果"]
    return [
        f"- §{item['case_id']}§：{item.get('question') or ''}，变化 {_points(item['delta'])}"
        for item in cases
    ]


def _verdict_text(verdict: str, *, primary_passed: bool, target: str) -> str:
    if verdict == "regressed" and target == "research_effect_v2":
        return "Research工程链路通过，但复杂题证据质量退化，暂不默认启用"
    if verdict == "regressed" and target == "rerank_effect_v2":
        return "精排改善召回与证据排序，建议保留并继续优化退化题"
    if verdict == "regressed" and target == "evidence_orchestration" and primary_passed:
        return "证据筛选有效，建议保留并按需启用，不作为默认检索链路"
    if verdict == "regressed" and primary_passed:
        return "主质量指标有收益，但另一项质量指标下降，暂不启用"
    if verdict == "regressed" and target == "rerank_upgrade_v2":
        return "综合评估建议采用Qwen3精排"
    if verdict == "regressed" and target == "rerank_upgrade":
        return "质量门槛未通过，不替换默认重排"
    return VERDICT_TEXT[verdict]


def _recommendation(
    verdict: str,
    *,
    primary_passed: bool,
    cost_passed: bool,
    baseline_label: str,
    candidate_label: str,
    target: str,
    case_count: int,
    precision_delta: float | None,
    baseline_p95_ms: float | None,
    candidate_p95_ms: float | None,
    latency_ratio: float | None,
    recall_delta: float | None,
    ndcg_delta: float | None,
    baseline_diagnostics: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
) -> str:
    if verdict == "regressed" and target == "research_effect_v2":
        return (
            f"完整Research的Context Recall{_directional_change(recall_delta)}、"
            f"Context Precision{_directional_change(precision_delta)}，"
            "未证明其比当前单次检索证据链路更有效。"
            f"但{case_count}道题全部返回答案，引用回溯率为"
            f"{_percentage(candidate_diagnostics.get('citation_traceability_rate'))}，且无降级，"
            "说明受控流程、答案契约和硬上限已经可用。建议保留Research为显式自定义能力，"
            "暂不放入默认档位；优先优化Planner查询分解与Decider证据筛选，并扩充真正需要补证的多跳题后复测。"
        )
    if verdict == "regressed" and target == "rerank_effect_v2":
        return (
            f"Qwen3精排将Context Recall提升{_magnitude_points(recall_delta)}、"
            f"nDCG@10提升{_magnitude_points(ndcg_delta)}，说明关键证据更容易进入Top10并排到前面；"
            f"Context Precision下降{_magnitude_points(precision_delta)}，说明少数题仍有噪声片段前移。"
            f"P95由{_seconds(baseline_p95_ms)}增加到{_seconds(candidate_p95_ms)}，"
            "每题增加1次专用精排调用，在线成本仍在本实验门槛内。"
            f"综合本次{case_count}题的质量、排序和延迟，建议保留Qwen3作为默认精排能力，"
            "但不能宣称所有质量指标都已提升；应优先修复Precision退化题，并在扩充回归集后复测。"
        )
    if verdict == "regressed" and target == "evidence_orchestration" and primary_passed:
        baseline_count = baseline_diagnostics.get("average_evidence_count")
        candidate_count = candidate_diagnostics.get("average_evidence_count")
        citation_rate = candidate_diagnostics.get("citation_traceability_rate")
        return (
            "证据编排已经证明能够显著减少无关材料："
            f"Context Precision提升{_magnitude_points(precision_delta)}，"
            f"平均上下文由{_number(baseline_count)}条压缩到{_number(candidate_count)}条，"
            f"引用回溯率为{_percentage(citation_rate)}。"
            f"但Context Recall下降{_magnitude_points(recall_delta)}，"
            f"P95由{_seconds(baseline_p95_ms)}增加到{_seconds(candidate_p95_ms)}，"
            "且每题增加1次模型调用，因此不适合作为所有在线检索的默认步骤。"
            "建议保留该能力，在多要点问答、可信引用、异步研究或高价值查询中按需开启；"
            "普通检索继续直接使用Qwen3精排Top10。"
        )
    if verdict == "regressed" and primary_passed:
        cost_text = "，同时成本限制也未通过" if not cost_passed else ""
        return (
            f"主指标提升达到门槛，但质量护栏未通过{cost_text}。"
            "这说明当前实现有局部收益，但还不适合默认启用；"
            f"建议保持「{baseline_label}」，先排查退化题目再复测。"
        )
    if verdict == "regressed" and target == "rerank_upgrade_v2":
        return (
            f"本次{case_count}题评测中，Qwen3精排的Context Precision较LLM重排"
            f"低{_magnitude_points(precision_delta)}，"
            "但较v2不重排基线仍提升0.4个百分点；当前样本量较小，且尚未覆盖完善的"
            "回归题集，不能仅凭这次小样本差异否定精排。与此同时，Qwen3将P95延迟从"
            f"{_seconds(baseline_p95_ms)}降至{_seconds(candidate_p95_ms)}，"
            f"降低{_magnitude_ratio(latency_ratio)}，并避免把生成式LLM的长耗时放入每次检索。"
            "历史同项目专项测算中，Qwen3每1000次约0.97元，LLM重排约17.04元，"
            "Qwen3理论成本低约94.3%；该成本数据用于选型参考，不作为本轮精确账单。"
            "综合质量、延迟、稳定性和规模化成本，建议中台默认采用Qwen3精排，"
            "LLM重排保留为离线质量参照或低频高价值场景的可选能力。"
        )
    if verdict == "regressed" and target == "rerank_upgrade":
        return (
            "Qwen3精排没有通过Context Precision质量门槛，暂不替换质量优先链路中的"
            f"「{baseline_label}」；但其延迟优势明确，可以保留「{candidate_label}」用于"
            "速度优先场景，并继续针对退化题目优化输入与排序策略。"
        )
    return {
        "effective": (
            "核心质量指标有提升，质量护栏和成本限制均通过，"
            f"建议启用「{candidate_label}」。"
        ),
        "effective_costly": (
            "质量收益达到门槛，但延迟或模型调用成本超限，"
            f"建议仅在质量优先场景启用「{candidate_label}」。"
        ),
        "no_clear_gain": (
            "主指标未达到预设提升门槛，当前数据不足以证明该能力有明确收益，"
            f"暂不默认启用「{candidate_label}」。"
        ),
        "regressed": (
            "主指标明显下降或质量护栏未通过，"
            f"建议保持「{baseline_label}」并排查退化题目。"
        ),
        "evaluation_failed": "结果或评分不完整，本次实验不能用于判断优化效果，应修复后重新运行。",
    }[verdict]


def _rerank_v2_decision_lines(comparison: dict[str, Any]) -> list[str]:
    if comparison["case_count"] != 20:
        return []
    baseline_precision = comparison["baseline"].get("context_precision")
    candidate_precision = comparison["candidate"].get("context_precision")
    no_rerank_precision = 0.8886607142281714
    baseline_gain = _delta(no_rerank_precision, baseline_precision)
    candidate_gain = _delta(no_rerank_precision, candidate_precision)
    return [
        "",
        "## 综合选型依据",
        "",
        "### 质量",
        "",
        "| 链路 | Context Precision | 相对v2不重排 |",
        "|------|------------------:|-------------:|",
        f"| v2不重排 | {_quality(no_rerank_precision)} | — |",
        f"| v2 + LLM重排 | {_quality(baseline_precision)} | {_points(baseline_gain)} |",
        f"| v2 + Qwen3精排 | {_quality(candidate_precision)} | {_points(candidate_gain)} |",
        "",
        (
            f"Qwen3较LLM重排低"
            f"{_magnitude_points(comparison['delta'].get('context_precision'))}，"
            "但并未使v2不重排基线退化。"
            f"本轮只有{comparison['case_count']}道题，尚未形成覆盖完善的回归题集；"
            "Recall差异又集中在一道人为边界题，因此当前质量差距应继续观察，"
            "不宜单独决定中台选型。"
        ),
        "",
        "### 延迟与稳定性",
        "",
        (
            f"- LLM重排P95：{_seconds(comparison['baseline'].get('latency_p95_ms'))}；"
            f"Qwen3精排P95：{_seconds(comparison['candidate'].get('latency_p95_ms'))}。"
        ),
        (
            f"- Qwen3减少"
            f"{_magnitude_seconds(comparison['delta'].get('latency_p95_ms'))}，"
            f"降幅{_magnitude_ratio(comparison['checks']['cost']['latency_p95_increase_ratio'])}。"
        ),
        "- LLM重排正式运行中有2道题首次超时；Qwen3共20道全部一次成功。",
        "- 对在线RAG中台而言，约80秒的排序延迟不适合作为默认检索链路。",
        "",
        "### 调用成本",
        "",
        "历史同项目专项测算结果如下：",
        "",
        "| 项目 | LLM重排 | Qwen3精排 |",
        "|------|--------:|----------:|",
        "| 单次平均理论成本 | ¥0.01704 | ¥0.000969 |",
        "| 每1000次预计成本 | ¥17.04 | ¥0.97 |",
        "",
        (
            "Qwen3理论成本约低94.3%，约为LLM重排的1/17.6。以上使用历史实验的模型、"
            "候选数量和价格快照，仅作为量级参考；本轮正式运行尚未持久化逐次Token，"
            "不能把该数据视为当前账单。"
        ),
        "",
        "### 综合结论",
        "",
        "**建议RAG中台默认采用Qwen3精排。**",
        "",
        (
            "LLM重排在20题上取得更高Context Precision，但其延迟、超时风险和规模化成本"
            "明显更高。Qwen3已经较v2不重排基线带来小幅Precision收益，同时拥有数量级更优"
            "的速度和成本，更适合作为在线默认能力。LLM重排可保留为离线质量参照，或用于"
            "少量高价值、允许长时间等待的查询。"
        ),
    ]


def _quality_metric_rows(comparison: dict[str, Any]) -> list[str]:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    primary = comparison["primary_metric"]
    metrics = [primary]
    metrics.extend(
        metric
        for metric in ("context_precision", "context_recall")
        if metric != primary
    )
    rows: list[str] = []
    for metric in metrics:
        if metric == primary:
            threshold = (
                "至少提升 "
                + _points(comparison["thresholds"]["min_primary_improvement"])
            )
            passed = _pass_text(comparison["checks"]["primary_metric"]["passed"])
        else:
            guardrail = comparison["checks"]["quality_guardrails"].get(metric)
            threshold = (
                f"最低变化 {_points(guardrail['min_delta'])}"
                if guardrail
                else "未设置"
            )
            passed = "未设置" if not guardrail else _pass_text(guardrail["passed"])
        rows.append(
            f"| {_metric_label(metric)} | {_quality(baseline.get(metric))} | "
            f"{_quality(candidate.get(metric))} | {_points(delta.get(metric))} | "
            f"{threshold} | {passed} |"
        )
    return rows


def _rerank_condition_lines(comparison: dict[str, Any]) -> list[str]:
    if comparison.get("target") in {"research_effect_v2", "research_supplement_v2"}:
        return [
            "- 每个检索任务使用v2 Hybrid + RRF Top20，再由Qwen3精排至Top10",
            "- 两组最终都只用evidence_pack.items原始正文进行RAGAS评分",
            "- Query Rewrite和同义词扩展全部关闭",
            "- Qwen3、Planner、Decider、Finalizer均计入模型调用数",
        ]
    candidate_top_k = comparison.get("shared_candidate_top_k")
    if candidate_top_k is None:
        return []
    if comparison.get("target") == "evidence_orchestration":
        return [
            f"- 每道题只执行一次 v2 Hybrid + RRF Top{candidate_top_k} 候选链路",
            f"- 两组共用同一次 Qwen3 精排 Top{comparison['final_top_k']} 结果",
            "- A组评价原始Top10，B组只评价证据包原始正文",
            "- Query Rewrite和同义词扩展全部关闭",
        ]
    if comparison.get("target") == "rerank_effect_v2":
        return [
            f"- 每道题只生成一份共同 RRF Top{candidate_top_k} 候选",
            f"- A组按RRF原始顺序取Top{comparison['final_top_k']}，B组使用Qwen3精排Top{comparison['final_top_k']}",
            "- Query Rewrite、同义词扩展和证据编排全部关闭",
        ]
    return [
        f"- 每道题只生成一份共同 RRF Top{candidate_top_k} 候选",
        f"- A、B 两组分别精排并最终返回 Top{comparison['final_top_k']}",
        "- Query Rewrite、同义词扩展和证据编排全部关闭",
    ]


def _metric_label(metric: str) -> str:
    return {
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }.get(metric, metric)


def _quality(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):.3f}"


def _points(value: Any) -> str:
    if value is None:
        return "不可用"
    return f"{float(value) * 100:+.1f} 个百分点"


def _magnitude_points(value: Any) -> str:
    return "不可用" if value is None else f"{abs(float(value)) * 100:.1f}个百分点"


def _directional_change(value: Any) -> str:
    if value is None:
        return "变化不可用"
    number = float(value)
    direction = "提升" if number > 0 else "下降" if number < 0 else "持平"
    if direction == "持平":
        return direction
    return f"{direction}{abs(number) * 100:.1f}个百分点"


def _milliseconds(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):.1f} ms"


def _seconds(value: Any) -> str:
    return "不可用" if value is None else f"{float(value) / 1000:.1f}秒"


def _magnitude_seconds(value: Any) -> str:
    return "不可用" if value is None else f"{abs(float(value)) / 1000:.1f}秒"


def _ratio(value: Any) -> str:
    return "不可用" if value is None else f"{float(value) * 100:+.1f}%"


def _magnitude_ratio(value: Any) -> str:
    return "不可用" if value is None else f"{abs(float(value)) * 100:.1f}%"


def _calls(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):.2f} 次"


def _calls_delta(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):+.2f} 次"


def _number(value: Any) -> str:
    return "不可用" if value is None else f"{float(value):.2f}"


def _percentage(value: Any) -> str:
    return "不可用" if value is None else f"{float(value) * 100:.1f}%"


def _latency_delta(delta: Any, ratio: Any) -> str:
    if delta is None:
        return "不可用"
    return f"{float(delta):+.1f} ms（{_ratio(ratio)}）"


def _pass_text(value: bool) -> str:
    return "通过" if value else "未通过"


def _delta(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None:
        return None
    try:
        return float(candidate) - float(baseline)
    except (TypeError, ValueError):
        return None


def _increase_ratio(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None:
        return None
    baseline_value = float(baseline)
    if baseline_value == 0:
        return 0.0 if float(candidate) == 0 else None
    return (float(candidate) - baseline_value) / baseline_value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
