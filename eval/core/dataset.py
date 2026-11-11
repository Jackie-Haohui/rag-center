from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from eval.core.models import EvalDataset


class DatasetValidationError(ValueError):
    pass


def load_dataset(path: Path) -> EvalDataset:
    if not path.is_file():
        raise DatasetValidationError(f"数据集不存在: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"数据集 JSON 无效: {exc}") from exc
    try:
        dataset = EvalDataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetValidationError(_format_validation_error(exc)) from exc
    _validate_known_golden(dataset)
    return dataset


def _validate_known_golden(dataset: EvalDataset) -> None:
    if not dataset.version.startswith("golden_basic_20"):
        return
    basic = dataset.cases_for_suite("basic")
    if len(basic) != 20:
        raise DatasetValidationError(
            f"{dataset.version} 的 basic suite 必须包含20道题，实际为{len(basic)}道"
        )
    expected_counts = {
        "direct_fact": 10,
        "paraphrase": 5,
        "multi_document": 5,
    }
    actual_counts = {
        case_type: sum(case.case_type == case_type for case in basic)
        for case_type in expected_counts
    }
    if actual_counts != expected_counts:
        raise DatasetValidationError(
            f"{dataset.version} 题型分布错误，预期{expected_counts}，实际{actual_counts}"
        )
    if "research" not in dataset.suites:
        return
    research = dataset.cases_for_suite("research")
    expected_research_counts = {
        "independent_multi_aspect": 4,
        "dependent_multi_hop": 4,
        "control": 2,
    }
    actual_research_counts = {
        case_type: sum(case.case_type == case_type for case in research)
        for case_type in expected_research_counts
    }
    if len(research) != 10 or actual_research_counts != expected_research_counts:
        raise DatasetValidationError(
            f"{dataset.version} 的research suite分布错误，"
            f"预期{expected_research_counts}，实际{actual_research_counts}"
        )
    for case in research:
        if not case.required_aspects:
            raise DatasetValidationError(f"Research题{case.id}缺少required_aspects")
        if case.expected_status is None:
            raise DatasetValidationError(f"Research题{case.id}缺少expected_status")
        if case.requires_multi_hop:
            if case.dependency is None or not case.dependency.target_chunk_ids:
                raise DatasetValidationError(f"多跳题{case.id}缺少补证目标Chunk")


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["数据集校验失败:"]
    for item in exc.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)
