from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault(
    "IR_DATASETS_HOME",
    str(Path(tempfile.gettempdir()) / "rag-center-ir-datasets"),
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "rag-center-numba-cache"),
)


class EvaluationScoringError(RuntimeError):
    pass


def score_run_directory(
    run_dir: Path,
    *,
    show_progress: bool = True,
    missing_only: bool = False,
    groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from eval.core.models import ExperimentConfig

    experiment = ExperimentConfig.model_validate(
        _read_json(run_dir / "experiment.snapshot.json")
    )
    if groups is None:
        groups = (
            ("baseline", "candidate")
            if missing_only
            or not experiment.baseline_run_dir
            or not (run_dir / "baseline.metrics.json").is_file()
            else ("candidate",)
        )
    rows_by_group = {
        group_name: _read_jsonl(run_dir / f"{group_name}.raw.jsonl")
        for group_name in groups
    }
    all_rows = [row for rows in rows_by_group.values() for row in rows]
    existing_cache = (
        load_existing_ragas_score_cache(run_dir, rows_by_group)
        if missing_only
        else {}
    )
    scoring_rows = (
        select_missing_ragas_rows(all_rows, existing_cache)
        if missing_only
        else all_rows
    )
    ragas_error: str | None = None
    try:
        new_cache = (
            compute_deduplicated_ragas_metrics(
                scoring_rows,
                max_workers=experiment.ragas_concurrency,
                timeout_seconds=experiment.timeout_seconds,
                show_progress=show_progress,
            )
            if scoring_rows
            else {}
        )
        ragas_score_cache = merge_ragas_score_caches(existing_cache, new_cache)
    except Exception as exc:  # noqa: BLE001 - scoring errors must become report data
        ragas_score_cache = existing_cache
        ragas_error = f"{type(exc).__name__}: {exc}"

    scored: dict[str, Any] = {}
    for group_name, rows in rows_by_group.items():
        metrics = score_group(
            rows,
            ragas_concurrency=experiment.ragas_concurrency,
            timeout_seconds=experiment.timeout_seconds,
            show_progress=show_progress,
            ragas_score_cache=ragas_score_cache,
            precomputed_ragas_error=ragas_error,
        )
        metrics["ragas"].update(
            {
                "missing_only": missing_only,
                "reused_existing_inputs": len(existing_cache),
                "retried_missing_inputs": len(scoring_rows),
            }
        )
        _write_json(run_dir / f"{group_name}.metrics.json", metrics)
        scored[group_name] = metrics
    return scored


def load_existing_ragas_score_cache(
    run_dir: Path,
    rows_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float | None]]:
    cache: dict[str, dict[str, float | None]] = {}
    for group_name, rows in rows_by_group.items():
        metrics_path = run_dir / f"{group_name}.metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _read_json(metrics_path)
        per_case = {
            item["case_id"]: item
            for item in metrics.get("per_case") or []
        }
        for row in rows:
            if row.get("error") is not None:
                continue
            saved = per_case.get(row["case_id"])
            if saved is None:
                continue
            key = ragas_input_key(row)
            cached = cache.setdefault(
                key,
                {"context_precision": None, "context_recall": None},
            )
            for metric in ("context_precision", "context_recall"):
                value = _maybe_float(saved.get(metric))
                if value is not None and cached[metric] is None:
                    cached[metric] = value
    return cache


def select_missing_ragas_rows(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("error") is not None or row.get("skip_ragas_reason"):
            continue
        key = ragas_input_key(row)
        if key in seen:
            continue
        seen.add(key)
        scores = cache.get(key) or {}
        if (
            scores.get("context_precision") is None
            or scores.get("context_recall") is None
        ):
            missing.append(row)
    return missing


def merge_ragas_score_caches(
    existing: dict[str, dict[str, float | None]],
    new: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    merged = {
        key: dict(scores)
        for key, scores in existing.items()
    }
    for key, scores in new.items():
        target = merged.setdefault(
            key,
            {"context_precision": None, "context_recall": None},
        )
        for metric in ("context_precision", "context_recall"):
            value = scores.get(metric)
            if value is not None and target.get(metric) is None:
                target[metric] = value
    return merged


def score_group(
    rows: list[dict[str, Any]],
    *,
    ragas_concurrency: int,
    timeout_seconds: float,
    show_progress: bool,
    ragas_score_cache: dict[str, dict[str, float | None]] | None = None,
    precomputed_ragas_error: str | None = None,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("error") is None]
    skipped = [row for row in successful if row.get("skip_ragas_reason")]
    scorable = [row for row in successful if not row.get("skip_ragas_reason")]
    failed = [row for row in rows if row.get("error") is not None]
    runtime = compute_runtime_metrics(successful, failed_count=len(failed))
    ranking_summary, ranking_per_case = compute_ranking_metrics(successful)
    ragas_error: str | None = None
    try:
        if precomputed_ragas_error is not None and ragas_score_cache is None:
            raise EvaluationScoringError(precomputed_ragas_error)
        if ragas_score_cache is None:
            ragas_summary, ragas_per_case = compute_ragas_metrics(
                scorable,
                max_workers=ragas_concurrency,
                timeout_seconds=timeout_seconds,
                show_progress=show_progress,
            )
        else:
            ragas_summary, ragas_per_case = ragas_metrics_from_cache(
                scorable,
                ragas_score_cache,
            )
            ragas_error = precomputed_ragas_error
    except Exception as exc:  # noqa: BLE001 - scoring errors must become report data
        ragas_summary = {
            "context_recall": None,
            "context_precision": None,
            "evaluated_cases": 0,
            "failed_cases": len(scorable),
        }
        ragas_per_case = []
        ragas_error = f"{type(exc).__name__}: {exc}"

    per_case_by_id = {item["case_id"]: item for item in ragas_per_case}
    ranking_by_id = {item["case_id"]: item for item in ranking_per_case}
    per_case: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "case_id": row["case_id"],
            "question": row["question"],
            "case_type": row.get("case_type"),
            "error": row.get("error"),
            "latency_ms": row.get("latency_ms"),
            "application_model_calls": row.get("application_model_calls"),
            "ragas_skipped": row.get("skip_ragas_reason"),
        }
        item.update(per_case_by_id.get(row["case_id"], {}))
        item.update(ranking_by_id.get(row["case_id"], {}))
        per_case.append(item)

    return {
        "summary": {
            **ragas_summary,
            "ragas_skipped_cases": len(skipped),
            **runtime,
            **ranking_summary,
        },
        "diagnostics": summarize_retrieval_diagnostics(successful),
        "per_case": per_case,
        "ragas": {
            "telemetry_disabled": os.getenv("RAGAS_DO_NOT_TRACK") == "true",
            "max_workers": ragas_concurrency,
            "timeout_seconds": timeout_seconds,
            "seed": 42,
            "deduplicated_identical_inputs": ragas_score_cache is not None,
            "error": ragas_error,
        },
    }


def compute_ranking_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    for row in rows:
        qrels = row.get("ranking_qrels")
        if not isinstance(qrels, dict) or not qrels:
            continue
        grades = [
            int(qrels.get(str(context.get("chunk_id") or ""), 0))
            for context in (row.get("contexts") or [])[:10]
        ]
        ideal = sorted((int(value) for value in qrels.values()), reverse=True)[:10]
        ideal_dcg = _dcg(ideal)
        per_case.append(
            {
                "case_id": row["case_id"],
                "ndcg_at_10": _dcg(grades) / ideal_dcg if ideal_dcg else None,
            }
        )
    values = [item["ndcg_at_10"] for item in per_case if item["ndcg_at_10"] is not None]
    return (
        {
            "ndcg_at_10": _mean_or_none(values),
            "rank_evaluated_cases": len(values),
        },
        per_case,
    )


def _dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def summarize_retrieval_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    affected_cases = 0
    injected_chunks = 0
    supplemental_chunks = 0
    evidence_counts: list[int] = []
    evidence_chars: list[int] = []
    citation_traceable = 0
    citation_total = 0
    evidence_statuses: dict[str, int] = {}
    answer_present_cases = 0
    answer_citations_valid_cases = 0
    degraded_cases = 0
    supplement_cases = 0
    rounds: list[int] = []
    retrieval_tasks: list[int] = []
    research_calls: list[int] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    for row in rows:
        diagnostics = row.get("diagnostics") or {}
        injected = int(diagnostics.get("graph_injected_count") or 0)
        supplemental = int(diagnostics.get("supplemental_chunk_count") or 0)
        injected_chunks += injected
        supplemental_chunks += supplemental
        if injected > 0 or supplemental > 0:
            affected_cases += 1
        if "evidence_count" in diagnostics:
            evidence_counts.append(int(diagnostics.get("evidence_count") or 0))
            evidence_chars.append(int(diagnostics.get("evidence_chars") or 0))
        citation_traceable += int(diagnostics.get("citation_traceable_items") or 0)
        citation_total += int(diagnostics.get("citation_total_items") or 0)
        status = diagnostics.get("evidence_status")
        if status:
            evidence_statuses[str(status)] = evidence_statuses.get(str(status), 0) + 1
        answer_present_cases += int(bool(diagnostics.get("answer_present")))
        answer_citations_valid_cases += int(bool(diagnostics.get("answer_citations_valid")))
        degraded_cases += int(bool(diagnostics.get("degraded")))
        supplement_cases += int(bool(diagnostics.get("supplement_executed")))
        if "round_count" in diagnostics:
            rounds.append(int(diagnostics.get("round_count") or 0))
            retrieval_tasks.append(int(diagnostics.get("retrieval_task_count") or 0))
            research_calls.append(int(diagnostics.get("research_llm_call_count") or 0))
            input_tokens.append(int(diagnostics.get("input_tokens") or 0))
            output_tokens.append(int(diagnostics.get("output_tokens") or 0))
    return {
        "graph_affected_cases": affected_cases,
        "graph_injected_chunks": injected_chunks,
        "supplemental_chunks": supplemental_chunks,
        "average_evidence_count": _mean_or_none(evidence_counts),
        "average_evidence_chars": _mean_or_none(evidence_chars),
        "citation_traceable_items": citation_traceable,
        "citation_total_items": citation_total,
        "citation_traceability_rate": (
            citation_traceable / citation_total if citation_total else None
        ),
        "evidence_statuses": evidence_statuses,
        "answer_present_cases": answer_present_cases,
        "answer_citations_valid_cases": answer_citations_valid_cases,
        "degraded_cases": degraded_cases,
        "supplement_cases": supplement_cases,
        "average_round_count": _mean_or_none(rounds),
        "average_retrieval_task_count": _mean_or_none(retrieval_tasks),
        "average_research_llm_call_count": _mean_or_none(research_calls),
        "average_input_tokens": _mean_or_none(input_tokens),
        "average_output_tokens": _mean_or_none(output_tokens),
    }


def compute_deduplicated_ragas_metrics(
    rows: list[dict[str, Any]],
    *,
    max_workers: int,
    timeout_seconds: float,
    show_progress: bool,
) -> dict[str, dict[str, float | None]]:
    unique_rows: list[dict[str, Any]] = []
    synthetic_id_to_key: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        if row.get("error") is not None or row.get("skip_ragas_reason"):
            continue
        key = ragas_input_key(row)
        if key in seen:
            continue
        seen.add(key)
        synthetic_id = f"ragas_input_{len(unique_rows)}"
        synthetic_row = {**row, "case_id": synthetic_id}
        unique_rows.append(synthetic_row)
        synthetic_id_to_key[synthetic_id] = key

    _, per_case = compute_ragas_metrics(
        unique_rows,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        show_progress=show_progress,
    )
    return {
        synthetic_id_to_key[item["case_id"]]: {
            "context_precision": item.get("context_precision"),
            "context_recall": item.get("context_recall"),
        }
        for item in per_case
    }


def ragas_metrics_from_cache(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, float | None]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    for row in rows:
        scores = cache.get(ragas_input_key(row), {})
        per_case.append(
            {
                "case_id": row["case_id"],
                "context_precision": scores.get("context_precision"),
                "context_recall": scores.get("context_recall"),
            }
        )
    recall_values = [
        item["context_recall"]
        for item in per_case
        if item["context_recall"] is not None
    ]
    precision_values = [
        item["context_precision"]
        for item in per_case
        if item["context_precision"] is not None
    ]
    evaluated_cases = sum(
        item["context_recall"] is not None and item["context_precision"] is not None
        for item in per_case
    )
    return (
        {
            "context_recall": _mean_or_none(recall_values),
            "context_precision": _mean_or_none(precision_values),
            "evaluated_cases": evaluated_cases,
            "failed_cases": len(rows) - evaluated_cases,
        },
        per_case,
    )


def ragas_input_key(row: dict[str, Any]) -> str:
    payload = [
        row.get("question"),
        row.get("ground_truth"),
        [context.get("text") for context in row.get("contexts") or []],
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def reconcile_run_identical_ragas_scores(run_dir: Path) -> int:
    baseline_rows = _read_jsonl(run_dir / "baseline.raw.jsonl")
    candidate_rows = _read_jsonl(run_dir / "candidate.raw.jsonl")
    baseline_metrics = _read_json(run_dir / "baseline.metrics.json")
    candidate_metrics = _read_json(run_dir / "candidate.metrics.json")
    reused = reconcile_identical_ragas_scores(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )
    if reused:
        _write_json(run_dir / "baseline.metrics.json", baseline_metrics)
        _write_json(run_dir / "candidate.metrics.json", candidate_metrics)
    return reused


def reconcile_identical_ragas_scores(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> int:
    baseline_rows_by_id = {row["case_id"]: row for row in baseline_rows}
    baseline_scores_by_id = {
        item["case_id"]: item for item in baseline_metrics.get("per_case") or []
    }
    reused = 0
    for candidate_row in candidate_rows:
        case_id = candidate_row["case_id"]
        baseline_row = baseline_rows_by_id.get(case_id)
        if (
            baseline_row is None
            or baseline_row.get("error") is not None
            or candidate_row.get("error") is not None
            or ragas_input_key(baseline_row) != ragas_input_key(candidate_row)
        ):
            continue
        baseline_scores = baseline_scores_by_id.get(case_id)
        candidate_scores = next(
            (
                item
                for item in candidate_metrics.get("per_case") or []
                if item["case_id"] == case_id
            ),
            None,
        )
        if baseline_scores is None or candidate_scores is None:
            continue
        candidate_scores["context_recall"] = baseline_scores.get("context_recall")
        candidate_scores["context_precision"] = baseline_scores.get("context_precision")
        reused += 1

    if not reused:
        return 0
    per_case = candidate_metrics.get("per_case") or []
    recall_values = [
        item["context_recall"]
        for item in per_case
        if item.get("context_recall") is not None
    ]
    precision_values = [
        item["context_precision"]
        for item in per_case
        if item.get("context_precision") is not None
    ]
    evaluated_cases = sum(
        item.get("context_recall") is not None
        and item.get("context_precision") is not None
        for item in per_case
    )
    candidate_metrics["summary"].update(
        {
            "context_recall": _mean_or_none(recall_values),
            "context_precision": _mean_or_none(precision_values),
            "evaluated_cases": evaluated_cases,
            "failed_cases": len(per_case) - evaluated_cases,
        }
    )
    baseline_metrics.setdefault("ragas", {})[
        "deduplicated_identical_inputs"
    ] = True
    candidate_metrics.setdefault("ragas", {}).update(
        {
            "deduplicated_identical_inputs": True,
            "reused_identical_cases": reused,
        }
    )
    return reused


def compute_ragas_metrics(
    rows: list[dict[str, Any]],
    *,
    max_workers: int,
    timeout_seconds: float,
    show_progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise EvaluationScoringError("没有成功的检索结果可供 RAGAS 评分")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from scripts.eval_utils import build_ragas_llm, load_ragas_metrics

    llm = build_ragas_llm()
    context_precision, context_recall = load_ragas_metrics()
    dataset = Dataset.from_dict(
        {
            "user_input": [row["question"] for row in rows],
            "reference": [row["ground_truth"] for row in rows],
            "retrieved_contexts": [
                [context["text"] for context in row.get("contexts") or []]
                for row in rows
            ],
        }
    )
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        run_config=RunConfig(
            max_workers=max_workers,
            timeout=timeout_seconds,
            max_retries=2,
            seed=42,
        ),
        show_progress=show_progress,
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    per_case: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scores = frame.iloc[index]
        per_case.append(
            {
                "case_id": row["case_id"],
                "context_precision": _maybe_float(scores.get("context_precision")),
                "context_recall": _maybe_float(scores.get("context_recall")),
            }
        )

    recall_values = [
        item["context_recall"]
        for item in per_case
        if item["context_recall"] is not None
    ]
    precision_values = [
        item["context_precision"]
        for item in per_case
        if item["context_precision"] is not None
    ]
    evaluated_cases = sum(
        item["context_recall"] is not None and item["context_precision"] is not None
        for item in per_case
    )
    return (
        {
            "context_recall": _mean_or_none(recall_values),
            "context_precision": _mean_or_none(precision_values),
            "evaluated_cases": evaluated_cases,
            "failed_cases": len(rows) - evaluated_cases,
        },
        per_case,
    )


def compute_runtime_metrics(
    successful_rows: list[dict[str, Any]],
    *,
    failed_count: int,
) -> dict[str, Any]:
    latencies = [
        float(row["latency_ms"])
        for row in successful_rows
        if row.get("latency_ms") is not None
    ]
    calls = [
        float(row["application_model_calls"])
        for row in successful_rows
        if row.get("application_model_calls") is not None
    ]
    return {
        "successful_cases": len(successful_rows),
        "request_failed_cases": failed_count,
        "latency_mean_ms": _mean_or_none(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": max(latencies) if latencies else None,
        "average_application_model_calls": _mean_or_none(calls),
        "model_calls_evaluated_cases": len(calls),
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


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
