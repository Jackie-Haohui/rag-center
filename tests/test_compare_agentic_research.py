from pathlib import Path

import pytest

from scripts import compare_agentic_research as evaluator


def test_prepared_dataset_has_four_reviewed_cases_per_type():
    dataset = evaluator.prepare_dataset()

    result = evaluator.validate_dataset(dataset, check_database=False)

    assert result["valid"] is True
    assert result["case_count"] == 16
    assert result["type_counts"] == {
        "simple": 4,
        "independent_multi_aspect": 4,
        "dependent_multi_hop": 4,
        "control": 4,
    }
    assert all(case["review_status"] == "reviewed" for case in dataset["cases"])


def test_four_case_smoke_selection_takes_one_from_each_type():
    selected = evaluator.balanced_cases(evaluator.prepare_dataset()["cases"], 4)

    assert {case["case_type"] for case in selected} == set(evaluator.CASE_TYPES)


def test_dependent_smoke_dataset_is_reviewed_and_isolated_from_formal_dataset():
    dataset = evaluator.prepare_smoke_dataset()

    result = evaluator.validate_dataset(dataset, check_database=False)

    assert result["valid"] is True
    assert result["case_count"] == 4
    assert result["type_counts"] == {"dependent_multi_hop": 4}
    assert dataset["evaluation_scope"] == "smoke"
    assert {case["id"] for case in dataset["cases"]}.isdisjoint(
        case["id"] for case in evaluator.prepare_dataset()["cases"]
    )


def test_case_type_selection_keeps_only_requested_type():
    cases = evaluator.prepare_dataset()["cases"]

    selected = evaluator.select_cases(cases, 2, ["dependent_multi_hop"])

    assert len(selected) == 2
    assert all(case["case_type"] == "dependent_multi_hop" for case in selected)


def test_round_chunk_ids_prefers_fused_candidates_over_anchor_tasks():
    record = {
        "rounds": [
            {
                "round": 1,
                "candidate_chunk_ids": ["anchor", "context-reference"],
                "tasks": [{"chunk_ids": ["anchor"]}],
            }
        ]
    }

    assert evaluator.round_chunk_ids(record, 1) == {"anchor", "context-reference"}


def test_round_chunk_ids_recovers_selected_context_evidence_from_older_results():
    record = {
        "rounds": [{"round": 1, "tasks": [{"chunk_ids": ["anchor"]}]}],
        "evidence_pack": {
            "items": [{"chunk_id": "context-reference", "rounds": [1]}]
        },
    }

    assert evaluator.round_chunk_ids(record, 1) == {"anchor", "context-reference"}


def test_global_lock_rejects_concurrent_group_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    first = evaluator.OutputLock(Path("a.json"))
    second = evaluator.OutputLock(Path("b.json"))

    with first, pytest.raises(FileExistsError), second:
        pass


def test_case_metrics_require_core_evidence_and_validate_references(monkeypatch):
    case = {
        "id": "case-1",
        "case_type": "simple",
        "kb_ids": [evaluator.KB_ID],
        "required_aspects": [
            {
                "aspect_id": "A1",
                "description": "规则",
                "acceptable_chunk_ids": ["chunk-1"],
            }
        ],
    }
    record = {
        "success": True,
        "degraded": False,
        "error": None,
        "answer": "结论[E1]",
        "evidence_pack": {
            "status": "complete",
            "groups": [
                {
                    "aspect": "规则",
                    "covered": True,
                    "evidence": [{"evidence_id": "E1", "role": "core"}],
                }
            ],
            "items": [{"evidence_id": "E1", "chunk_id": "chunk-1"}],
        },
        "metadata": {
            "initial_evidence_status": "complete",
            "stop_reason": "evidence_complete",
            "latency_ms": 8,
            "round_count": 1,
            "retrieval_task_count": 1,
            "llm_call_count": 3,
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }
    monkeypatch.setattr(evaluator, "traceable_ids", lambda ids, case: ids)

    row = evaluator.evaluate_case(case, record)

    assert row["complete"] is True
    assert row["evidence_precision"] == 1.0
    assert row["status_match"] is True
    assert row["citation_traceability"] == 1.0
    assert row["answer_references_valid"] is True


def test_supporting_evidence_does_not_cover_required_aspect(monkeypatch):
    case = {
        "id": "case-1",
        "case_type": "simple",
        "kb_ids": [evaluator.KB_ID],
        "required_aspects": [
            {
                "aspect_id": "A1",
                "description": "规则",
                "acceptable_chunk_ids": ["chunk-1"],
            }
        ],
    }
    record = {
        "success": True,
        "degraded": False,
        "evidence_pack": {
            "status": "missing",
            "groups": [
                {
                    "aspect": "规则",
                    "covered": False,
                    "evidence": [{"evidence_id": "E1", "role": "supporting"}],
                }
            ],
            "items": [{"evidence_id": "E1", "chunk_id": "chunk-1"}],
        },
        "metadata": {},
    }
    monkeypatch.setattr(evaluator, "traceable_ids", lambda ids, case: ids)

    row = evaluator.evaluate_case(case, record)

    assert row["covered_aspects"] == 0
    assert row["actual_status"] == "missing"


def test_no_query_or_repair_gain_is_not_a_conditional_pass():
    checks = {
        "independent_coverage_b_gt_a": False,
        "independent_complete_b_adds_one": False,
        "simple_regression_within_5pp": True,
        "dependent_complete_c_gt_b": False,
        "dependent_complete_c_adds_one": False,
        "multi_hop_repair_evaluable": True,
        "multi_hop_repair_at_least_50pct": False,
        "c_coverage_not_below_b": True,
        "c_precision_at_least_80pct": False,
        "c_precision_drop_within_5pp": True,
        "c_status_accuracy_at_least_80pct": False,
        "citation_traceability_100pct": True,
        "unnecessary_supplement_at_most_10pct": True,
        "agent_hard_limits": True,
    }

    assert evaluator.decide_conclusion(checks) == "不通过"
