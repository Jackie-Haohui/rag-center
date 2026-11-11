from pathlib import Path

from eval.core.dataset import load_dataset
from eval.core.experiment import load_experiment
from eval.core.report import build_comparison, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = load_dataset(PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json")
EXPERIMENT = load_experiment(
    PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json",
    project_root=PROJECT_ROOT,
)
RERANK_EXPERIMENT = load_experiment(
    PROJECT_ROOT / "eval" / "experiments" / "rerank_upgrade.json",
    project_root=PROJECT_ROOT,
)
CONTEXT_GRAPH_EXPERIMENT = load_experiment(
    PROJECT_ROOT / "eval" / "experiments" / "context_graph_upgrade.json",
    project_root=PROJECT_ROOT,
)
RESEARCH_EXPERIMENT = load_experiment(
    PROJECT_ROOT / "eval" / "experiments" / "research_effect_v2.json",
    project_root=PROJECT_ROOT,
)


def _metrics(
    *,
    recall: float,
    precision: float,
    p95: float,
    calls: float,
    failed: int = 0,
) -> dict:
    return {
        "summary": {
            "context_recall": recall,
            "context_precision": precision,
            "latency_p95_ms": p95,
            "average_application_model_calls": calls,
            "request_failed_cases": failed,
            "failed_cases": failed,
        },
        "per_case": [],
        "ragas": {"error": None},
    }


def test_effective_verdict_and_explicit_markdown_conclusion():
    comparison = build_comparison(
        experiment=EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.70, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.76, precision=0.79, p95=950, calls=1),
    )
    markdown = render_markdown(comparison)

    assert comparison["verdict"] == "effective"
    assert "**有效，建议启用。**" in markdown
    assert "+6.0 个百分点" in markdown
    assert "平均应用模型调用数" in markdown


def test_no_clear_gain_verdict():
    comparison = build_comparison(
        experiment=EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.70, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.71, precision=0.80, p95=820, calls=1),
    )

    assert comparison["verdict"] == "no_clear_gain"


def test_zero_primary_threshold_accepts_guardrail_quality_improvement():
    experiment = EXPERIMENT.model_copy(deep=True)
    experiment.decision.min_primary_improvement = 0.0
    comparison = build_comparison(
        experiment=experiment,
        dataset=DATASET,
        baseline=_metrics(recall=0.80, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.80, precision=0.81, p95=800, calls=0),
    )

    assert comparison["verdict"] == "effective"
    assert comparison["checks"]["quality_improved"] is True


def test_zero_primary_threshold_does_not_accept_unchanged_quality():
    experiment = EXPERIMENT.model_copy(deep=True)
    experiment.decision.min_primary_improvement = 0.0
    comparison = build_comparison(
        experiment=experiment,
        dataset=DATASET,
        baseline=_metrics(recall=0.80, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.80, precision=0.80, p95=800, calls=0),
    )

    assert comparison["verdict"] == "no_clear_gain"


def test_zero_primary_threshold_rejects_primary_regression():
    experiment = EXPERIMENT.model_copy(deep=True)
    experiment.decision.min_primary_improvement = 0.0
    comparison = build_comparison(
        experiment=experiment,
        dataset=DATASET,
        baseline=_metrics(recall=0.80, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.79, precision=0.81, p95=800, calls=0),
    )

    assert comparison["verdict"] == "regressed"


def test_quality_guardrail_failure_is_regression():
    comparison = build_comparison(
        experiment=EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.70, precision=0.80, p95=800, calls=0),
        candidate=_metrics(recall=0.76, precision=0.75, p95=820, calls=1),
    )

    assert comparison["verdict"] == "regressed"
    assert comparison["verdict_text"] == "主质量指标有收益，但另一项质量指标下降，暂不启用"
    assert "有局部收益" in comparison["recommendation"]


def test_rerank_report_contains_only_four_formal_metrics():
    comparison = build_comparison(
        experiment=RERANK_EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.80, precision=0.78, p95=30000, calls=1),
        candidate=_metrics(recall=0.80, precision=0.82, p95=1200, calls=1),
    )

    markdown = render_markdown(comparison)
    result_table = markdown.split("## 核心结果", 1)[1].split("## 实验目标", 1)[0]

    assert comparison["verdict"] == "effective"
    assert result_table.count("| Context Precision |") == 1
    assert result_table.count("| Context Recall |") == 1
    assert result_table.count("| P95 延迟 |") == 1
    assert result_table.count("| 平均应用模型调用数 |") == 1
    assert "nDCG" not in markdown
    assert "MRR" not in markdown
    assert "共同 RRF Top20" in markdown
    assert "最终返回 Top10" in markdown
    assert "Context Precision 从 0.780 变为 0.820" in markdown


def test_regressed_rerank_keeps_qwen_for_speed_profile():
    comparison = build_comparison(
        experiment=RERANK_EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.90, precision=0.94, p95=80000, calls=1),
        candidate=_metrics(recall=0.93, precision=0.89, p95=3300, calls=1),
    )

    assert comparison["verdict"] == "regressed"
    assert comparison["verdict_text"] == "质量门槛未通过，不替换默认重排"
    assert "速度优先场景" in comparison["recommendation"]


def test_v2_rerank_report_recommends_qwen_from_overall_tradeoff():
    experiment = load_experiment(
        PROJECT_ROOT / "eval" / "experiments" / "rerank_upgrade_v2.json",
        project_root=PROJECT_ROOT,
    )
    comparison = build_comparison(
        experiment=experiment,
        dataset=DATASET,
        baseline=_metrics(recall=0.95, precision=0.943, p95=79684, calls=1),
        candidate=_metrics(recall=0.983, precision=0.892, p95=3279, calls=1),
    )
    markdown = render_markdown(comparison)

    assert comparison["verdict"] == "regressed"
    assert comparison["verdict_text"] == "综合评估建议采用Qwen3精排"
    assert "每1000次预计成本" in markdown
    assert "默认采用Qwen3精排" in markdown


def test_case_change_sections_exclude_unchanged_cases():
    baseline = _metrics(recall=0.70, precision=0.80, p95=800, calls=0)
    candidate = _metrics(recall=0.76, precision=0.79, p95=820, calls=1)
    baseline["per_case"] = [
        {"case_id": "better", "question": "改善", "context_recall": 0.0},
        {"case_id": "same", "question": "不变", "context_recall": 1.0},
        {"case_id": "worse", "question": "退化", "context_recall": 1.0},
    ]
    candidate["per_case"] = [
        {"case_id": "better", "question": "改善", "context_recall": 1.0},
        {"case_id": "same", "question": "不变", "context_recall": 1.0},
        {"case_id": "worse", "question": "退化", "context_recall": 0.0},
    ]

    comparison = build_comparison(
        experiment=EXPERIMENT,
        dataset=DATASET,
        baseline=baseline,
        candidate=candidate,
    )

    assert [item["case_id"] for item in comparison["case_changes"]["improved"]] == [
        "better"
    ]
    assert [item["case_id"] for item in comparison["case_changes"]["regressed"]] == [
        "worse"
    ]


def test_context_graph_report_shows_index_versions_and_diagnostics():
    baseline = _metrics(recall=0.70, precision=0.80, p95=800, calls=0)
    candidate = _metrics(recall=0.75, precision=0.79, p95=900, calls=0)
    baseline["diagnostics"] = {
        "graph_affected_cases": 0,
        "graph_injected_chunks": 0,
        "supplemental_chunks": 0,
    }
    candidate["diagnostics"] = {
        "graph_affected_cases": 4,
        "graph_injected_chunks": 7,
        "supplemental_chunks": 9,
    }

    comparison = build_comparison(
        experiment=CONTEXT_GRAPH_EXPERIMENT,
        dataset=DATASET,
        baseline=baseline,
        candidate=candidate,
    )
    markdown = render_markdown(comparison)

    assert "索引版本：baseline=v1，candidate=v2" in markdown
    assert "发生图谱注入或上下文补充的题目：4 道" in markdown
    assert "以上仅用于解释结果，不参与正式验收门槛" in markdown


def test_report_uses_executed_case_count_for_limited_run():
    baseline = _metrics(recall=0.70, precision=0.80, p95=800, calls=0)
    candidate = _metrics(recall=0.75, precision=0.79, p95=900, calls=0)
    baseline["summary"]["successful_cases"] = 2
    candidate["summary"]["successful_cases"] = 2

    comparison = build_comparison(
        experiment=CONTEXT_GRAPH_EXPERIMENT,
        dataset=DATASET,
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison["case_count"] == 2
    assert "的 2 道题上" in render_markdown(comparison)


def test_incomplete_results_cannot_claim_effectiveness():
    comparison = build_comparison(
        experiment=EXPERIMENT,
        dataset=DATASET,
        baseline=_metrics(recall=0.70, precision=0.80, p95=800, calls=0),
        candidate=_metrics(
            recall=0.80,
            precision=0.82,
            p95=820,
            calls=1,
            failed=1,
        ),
    )

    assert comparison["verdict"] == "evaluation_failed"


def test_research_quality_uses_same_evaluable_case_intersection():
    baseline = _metrics(recall=0.5, precision=0.5, p95=800, calls=2)
    candidate = _metrics(recall=0.5, precision=0.5, p95=900, calls=5)
    baseline["summary"].update({"successful_cases": 2, "ragas_skipped_cases": 1})
    candidate["summary"].update({"successful_cases": 2, "ragas_skipped_cases": 0})
    baseline["diagnostics"] = {
        "citation_traceability_rate": 1.0,
        "answer_present_cases": 0,
        "answer_citations_valid_cases": 0,
        "degraded_cases": 0,
    }
    candidate["diagnostics"] = {
        "citation_traceability_rate": 1.0,
        "answer_present_cases": 2,
        "answer_citations_valid_cases": 2,
        "degraded_cases": 0,
    }
    baseline["per_case"] = [
        {
            "case_id": "answerable",
            "context_precision": 0.8,
            "context_recall": 1.0,
        },
        {
            "case_id": "expected_missing",
            "ragas_skipped": "expected_missing_no_context",
        },
    ]
    candidate["per_case"] = [
        {
            "case_id": "answerable",
            "context_precision": 0.9,
            "context_recall": 1.0,
        },
        {
            "case_id": "expected_missing",
            "context_precision": 0.0,
            "context_recall": 0.0,
        },
    ]

    comparison = build_comparison(
        experiment=RESEARCH_EXPERIMENT,
        dataset=DATASET,
        baseline=baseline,
        candidate=candidate,
    )
    markdown = render_markdown(comparison)

    assert comparison["baseline"]["context_precision"] == 0.8
    assert comparison["candidate"]["context_precision"] == 0.9
    assert comparison["baseline"]["context_recall"] == 1.0
    assert comparison["candidate"]["context_recall"] == 1.0
    assert comparison["quality_comparison"]["case_count"] == 1
    assert "Context指标共同可评价题目：1/2 道" in markdown
    assert "expected_missing_no_context" in markdown
