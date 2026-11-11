import pytest

from eval.core.metrics import (
    compute_ranking_metrics,
    compute_runtime_metrics,
    merge_ragas_score_caches,
    percentile,
    ragas_input_key,
    ragas_metrics_from_cache,
    reconcile_identical_ragas_scores,
    select_missing_ragas_rows,
)


def test_percentile_and_runtime_metrics():
    successful = [
        {"latency_ms": 100, "application_model_calls": 1},
        {"latency_ms": 200, "application_model_calls": 2},
        {"latency_ms": 300, "application_model_calls": 3},
    ]

    metrics = compute_runtime_metrics(successful, failed_count=1)

    assert percentile([100, 200, 300], 50) == 200
    assert metrics["successful_cases"] == 3
    assert metrics["request_failed_cases"] == 1
    assert metrics["latency_p95_ms"] == pytest.approx(290)
    assert metrics["average_application_model_calls"] == 2


def test_ndcg_at_10_rewards_putting_strong_evidence_first():
    qrels = {"strong": 3, "supporting": 2, "noise": 0}
    better = {
        "case_id": "better",
        "ranking_qrels": qrels,
        "contexts": [
            {"chunk_id": "strong"},
            {"chunk_id": "supporting"},
            {"chunk_id": "noise"},
        ],
    }
    worse = {
        "case_id": "worse",
        "ranking_qrels": qrels,
        "contexts": [
            {"chunk_id": "noise"},
            {"chunk_id": "supporting"},
            {"chunk_id": "strong"},
        ],
    }

    _, per_case = compute_ranking_metrics([better, worse])

    assert per_case[0]["ndcg_at_10"] == pytest.approx(1.0)
    assert per_case[0]["ndcg_at_10"] > per_case[1]["ndcg_at_10"]


def test_identical_ragas_inputs_reuse_the_same_score():
    baseline = {
        "case_id": "same_case",
        "question": "同一个问题",
        "ground_truth": "同一个答案",
        "contexts": [{"text": "同一个片段", "score": 0.8}],
    }
    candidate = {
        **baseline,
        "contexts": [{"text": "同一个片段", "score": 0.9}],
    }
    cache = {
        ragas_input_key(baseline): {
            "context_recall": 1.0,
            "context_precision": 0.8,
        }
    }

    summary, per_case = ragas_metrics_from_cache([candidate], cache)

    assert ragas_input_key(baseline) == ragas_input_key(candidate)
    assert summary["context_recall"] == 1.0
    assert summary["context_precision"] == 0.8
    assert per_case[0]["context_recall"] == 1.0


def test_changed_context_order_is_a_distinct_ragas_input():
    first = {
        "question": "问题",
        "ground_truth": "答案",
        "contexts": [{"text": "片段一"}, {"text": "片段二"}],
    }
    second = {
        **first,
        "contexts": list(reversed(first["contexts"])),
    }

    assert ragas_input_key(first) != ragas_input_key(second)


def test_reconcile_identical_contexts_uses_baseline_score():
    baseline_row = {
        "case_id": "same_case",
        "question": "问题",
        "ground_truth": "答案",
        "contexts": [{"text": "完全相同的片段"}],
        "error": None,
    }
    candidate_row = {**baseline_row, "group": "candidate"}
    baseline_metrics = {
        "summary": {},
        "per_case": [
            {
                "case_id": "same_case",
                "context_recall": 1.0,
                "context_precision": 0.8,
            }
        ],
        "ragas": {},
    }
    candidate_metrics = {
        "summary": {},
        "per_case": [
            {
                "case_id": "same_case",
                "context_recall": 0.0,
                "context_precision": 0.2,
            }
        ],
        "ragas": {},
    }

    reused = reconcile_identical_ragas_scores(
        baseline_rows=[baseline_row],
        candidate_rows=[candidate_row],
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    assert reused == 1
    assert candidate_metrics["summary"]["context_recall"] == 1.0
    assert candidate_metrics["summary"]["context_precision"] == 0.8
    assert candidate_metrics["per_case"][0]["context_recall"] == 1.0
    assert candidate_metrics["ragas"]["reused_identical_cases"] == 1


def test_missing_only_scoring_reuses_successes_and_selects_only_gaps():
    complete = {
        "case_id": "complete",
        "question": "完整问题",
        "ground_truth": "完整答案",
        "contexts": [{"text": "完整片段"}],
        "error": None,
    }
    missing = {
        "case_id": "missing",
        "question": "缺失问题",
        "ground_truth": "缺失答案",
        "contexts": [{"text": "缺失片段"}],
        "error": None,
    }
    complete_key = ragas_input_key(complete)
    missing_key = ragas_input_key(missing)
    existing = {
        complete_key: {"context_precision": 0.8, "context_recall": 1.0},
        missing_key: {"context_precision": 0.7, "context_recall": None},
    }

    selected = select_missing_ragas_rows([complete, missing], existing)
    merged = merge_ragas_score_caches(
        existing,
        {
            missing_key: {"context_precision": 0.1, "context_recall": 0.9},
        },
    )

    assert [row["case_id"] for row in selected] == ["missing"]
    assert merged[complete_key] == existing[complete_key]
    assert merged[missing_key]["context_precision"] == 0.7
    assert merged[missing_key]["context_recall"] == 0.9
