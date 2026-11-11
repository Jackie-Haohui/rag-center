from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from eval.core.dataset import DatasetValidationError, load_dataset
from eval.core.models import ExperimentConfig


class ExperimentValidationError(ValueError):
    pass


def load_experiment(path: Path, *, project_root: Path | None = None) -> ExperimentConfig:
    if not path.is_file():
        raise ExperimentValidationError(f"实验配置不存在: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(f"实验配置 JSON 无效: {exc}") from exc
    try:
        experiment = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ExperimentValidationError(_format_validation_error(exc)) from exc

    root = project_root or Path.cwd()
    dataset_path = (root / experiment.dataset).resolve()
    try:
        dataset = load_dataset(dataset_path)
    except DatasetValidationError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    if experiment.suite not in dataset.suites:
        raise ExperimentValidationError(
            f"实验 suite {experiment.suite!r} 不存在于数据集 {dataset.version}"
        )
    return experiment


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["实验配置校验失败:"]
    for item in exc.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)
