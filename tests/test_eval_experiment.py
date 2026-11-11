import json
from pathlib import Path

import pytest

from eval.core.experiment import ExperimentValidationError, load_experiment
from eval.core.models import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "query_rewrite.json"
HYBRID_EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "hybrid_search.json"
RERANK_EXPERIMENT_PATH = PROJECT_ROOT / "eval" / "experiments" / "rerank_upgrade.json"
RERANK_V2_EXPERIMENT_PATH = (
    PROJECT_ROOT / "eval" / "experiments" / "rerank_upgrade_v2.json"
)
RERANK_EFFECT_V2_EXPERIMENT_PATH = (
    PROJECT_ROOT / "eval" / "experiments" / "rerank_effect_v2.json"
)
CONTEXT_GRAPH_EXPERIMENT_PATH = (
    PROJECT_ROOT / "eval" / "experiments" / "context_graph_upgrade.json"
)


def _experiment_payload() -> dict:
    return json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))


def test_query_rewrite_experiment_is_valid():
    experiment = load_experiment(EXPERIMENT_PATH, project_root=PROJECT_ROOT)

    assert experiment.changed_fields == ["query_options.enabled"]
    assert experiment.baseline.query_options["enabled"] is False
    assert experiment.candidate.query_options["enabled"] is True


def test_hybrid_experiment_uses_tuned_keyword_candidates_only():
    experiment = load_experiment(HYBRID_EXPERIMENT_PATH, project_root=PROJECT_ROOT)

    assert experiment.changed_fields == ["retrieval_options.mode"]
    assert experiment.baseline.retrieval_options["mode"] == "vector"
    assert experiment.candidate.retrieval_options["mode"] == "hybrid"
    assert experiment.candidate.retrieval_options["bm25_top_k"] == 5
    assert experiment.candidate.retrieval_options["rrf_k"] == 10
    assert experiment.candidate.query_options["enabled"] is False
    assert experiment.candidate.query_options["synonym_enabled"] is False
    assert experiment.candidate.rerank_options["enabled"] is False


def test_rerank_experiment_freezes_shared_top20_and_final_top10():
    experiment = load_experiment(RERANK_EXPERIMENT_PATH, project_root=PROJECT_ROOT)

    assert experiment.changed_fields == ["rerank_provider"]
    assert experiment.shared_candidate_top_k == 20
    assert experiment.baseline.rerank_provider == "llm"
    assert experiment.candidate.rerank_provider == "qwen3"
    assert experiment.baseline.top_k == experiment.candidate.top_k == 10
    assert experiment.baseline.retrieval_options["mode"] == "hybrid"
    assert experiment.baseline.query_options["enabled"] is False
    assert experiment.baseline.query_options["synonym_enabled"] is False
    assert experiment.baseline.evidence_options["enabled"] is False


def test_v2_rerank_experiment_uses_v2_for_both_groups():
    experiment = load_experiment(RERANK_V2_EXPERIMENT_PATH, project_root=PROJECT_ROOT)

    assert experiment.changed_fields == ["rerank_provider"]
    assert experiment.shared_candidate_top_k == 20
    assert experiment.baseline.index_version == "v2"
    assert experiment.candidate.index_version == "v2"
    assert experiment.baseline.rerank_provider == "llm"
    assert experiment.candidate.rerank_provider == "qwen3"
    assert experiment.baseline.top_k == experiment.candidate.top_k == 10


def test_v2_rerank_effect_experiment_compares_rrf_with_qwen3():
    experiment = load_experiment(
        RERANK_EFFECT_V2_EXPERIMENT_PATH,
        project_root=PROJECT_ROOT,
    )

    assert experiment.changed_fields == ["rerank_options.enabled", "rerank_provider"]
    assert experiment.shared_candidate_top_k == 20
    assert experiment.baseline.rerank_provider == "noop"
    assert experiment.candidate.rerank_provider == "qwen3"
    assert experiment.baseline.rerank_options["enabled"] is False
    assert experiment.candidate.rerank_options["enabled"] is True
    assert experiment.baseline.index_version == experiment.candidate.index_version == "v2"


def test_context_graph_experiment_compares_v1_and_v2_without_rerank():
    experiment = load_experiment(CONTEXT_GRAPH_EXPERIMENT_PATH, project_root=PROJECT_ROOT)

    assert experiment.changed_fields == ["index_version"]
    assert experiment.baseline.index_version == "v1"
    assert experiment.candidate.index_version == "v2"
    assert experiment.baseline.top_k == experiment.candidate.top_k == 10
    assert experiment.baseline.retrieval_options["vector_top_k"] == 20
    assert experiment.candidate.retrieval_options["bm25_top_k"] == 20
    assert experiment.baseline.rerank_options["enabled"] is False
    assert experiment.candidate.rerank_options["enabled"] is False
    assert experiment.baseline.query_options["synonym_enabled"] is False
    assert experiment.candidate.evidence_options["enabled"] is False


def test_undeclared_group_difference_is_rejected():
    payload = _experiment_payload()
    payload["candidate"]["top_k"] = 8

    with pytest.raises(ValueError, match="未声明差异"):
        ExperimentConfig.model_validate(payload)


def test_declared_but_unchanged_field_is_rejected():
    payload = _experiment_payload()
    payload["candidate"]["query_options"]["enabled"] = False

    with pytest.raises(ValueError, match="实际没有变化"):
        ExperimentConfig.model_validate(payload)


def test_extra_experiment_group_is_rejected():
    payload = _experiment_payload()
    payload["third_group"] = payload["candidate"]

    with pytest.raises(ValueError, match="Extra inputs"):
        ExperimentConfig.model_validate(payload)


def test_unknown_suite_is_rejected(tmp_path: Path):
    payload = _experiment_payload()
    payload["suite"] = "missing"
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ExperimentValidationError, match="不存在于数据集"):
        load_experiment(path, project_root=PROJECT_ROOT)
