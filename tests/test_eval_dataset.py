import json
from pathlib import Path

import pytest

from eval.core.dataset import DatasetValidationError, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_basic_20.json"


def _dataset_payload() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _write_dataset(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_golden_basic_20_is_valid():
    dataset = load_dataset(DATASET_PATH)

    assert dataset.version == "golden_basic_20_v2"
    basic = dataset.cases_for_suite("basic")
    assert len(basic) == 20
    assert sum(case.case_type == "direct_fact" for case in basic) == 10
    assert sum(case.case_type == "paraphrase" for case in basic) == 5
    assert sum(case.case_type == "multi_document" for case in basic) == 5
    assert sum("lexical" in case.tags for case in basic) == 4


def test_duplicate_case_id_is_rejected(tmp_path: Path):
    payload = _dataset_payload()
    payload["version"] = "test"
    payload["cases"][1]["id"] = payload["cases"][0]["id"]

    with pytest.raises(DatasetValidationError, match="case ID 重复"):
        load_dataset(_write_dataset(tmp_path, payload))


def test_empty_ground_truth_is_rejected(tmp_path: Path):
    payload = _dataset_payload()
    payload["version"] = "test"
    payload["cases"][0]["ground_truth"] = ""

    with pytest.raises(DatasetValidationError, match="ground_truth"):
        load_dataset(_write_dataset(tmp_path, payload))


def test_unknown_kb_alias_is_rejected(tmp_path: Path):
    payload = _dataset_payload()
    payload["version"] = "test"
    payload["cases"][0]["kb_alias"] = "missing"

    with pytest.raises(DatasetValidationError, match="未知知识库别名"):
        load_dataset(_write_dataset(tmp_path, payload))


def test_suite_unknown_case_is_rejected(tmp_path: Path):
    payload = _dataset_payload()
    payload["version"] = "test"
    payload["suites"]["basic"]["case_ids"].append("missing_case")

    with pytest.raises(DatasetValidationError, match="不存在的 case"):
        load_dataset(_write_dataset(tmp_path, payload))


def test_invalid_case_type_is_rejected(tmp_path: Path):
    payload = _dataset_payload()
    payload["version"] = "test"
    payload["cases"][0]["case_type"] = "unknown"

    with pytest.raises(DatasetValidationError, match="case_type"):
        load_dataset(_write_dataset(tmp_path, payload))


def test_known_golden_requires_exact_case_distribution(tmp_path: Path):
    payload = _dataset_payload()
    payload["suites"]["basic"]["case_ids"].pop()

    with pytest.raises(DatasetValidationError, match="必须包含20道题"):
        load_dataset(_write_dataset(tmp_path, payload))
