from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from app.core.index_versions import LEGACY_INDEX_VERSION
from eval.core.models import EvalCase, EvalDataset, ExperimentConfig, ExperimentGroup

EVAL_USER_ID = "eval_runner"


class EvaluationRunError(RuntimeError):
    pass


def _experiment_snapshot(experiment: ExperimentConfig) -> dict[str, Any]:
    """Serialize a reloadable experiment snapshot.

    Named profiles are expanded to production presets during validation.  Their
    advanced fields must not be persisted, otherwise loading the snapshot would
    incorrectly treat the generated preset values as user overrides.
    """
    snapshot = experiment.model_dump(mode="json")
    advanced_fields = {
        "top_k",
        "retrieval_options",
        "rerank_options",
        "evidence_options",
        "query_options",
    }
    for group_name in ("baseline", "candidate"):
        group = snapshot[group_name]
        if group.get("profile") in {"speed", "balanced", "quality"}:
            for field_name in advanced_fields:
                group.pop(field_name, None)
    return snapshot


async def run_retrieval_groups(
    *,
    experiment: ExperimentConfig,
    dataset: EvalDataset,
    api_key: str,
    base_url: str,
    output_root: Path,
    limit: int | None = None,
    case_ids: list[str] | None = None,
    output_suffix: str | None = None,
) -> Path:
    cases = dataset.cases_for_suite(experiment.suite)
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case.id in requested]
        missing = requested - {case.id for case in cases}
        if missing:
            raise EvaluationRunError("suite中不存在指定题目: " + ", ".join(sorted(missing)))
    if limit is not None:
        if limit < 1:
            raise EvaluationRunError("--limit 必须大于0")
        cases = cases[:limit]
    if not cases:
        raise EvaluationRunError("当前 suite 没有可运行题目")

    run_dir = _create_run_dir(
        output_root=output_root,
        experiment_id=experiment.experiment_id,
        output_suffix=output_suffix,
    )
    _write_json(run_dir / "experiment.snapshot.json", _experiment_snapshot(experiment))
    _write_json(run_dir / "dataset.snapshot.json", dataset.model_dump(mode="json"))

    started_at = datetime.now(UTC)
    manifest = _build_manifest(
        experiment=experiment,
        dataset=dataset,
        started_at=started_at,
        case_count=len(cases),
        status="running",
    )
    _write_json(run_dir / "manifest.json", manifest)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(experiment.timeout_seconds)
    if experiment.baseline_run_dir and experiment.target in {
        "evidence_orchestration",
        "rerank_effect_v2",
        "research_effect_v2",
        "research_supplement_v2",
    }:
        raise EvaluationRunError("当前专题实验不支持冻结 baseline 复用")
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
        trust_env=False,
    ) as client:
        if experiment.target in {"research_effect_v2", "research_supplement_v2"}:
            group_results, shared_pipeline = await _run_research_cases_bounded(
                client=client,
                dataset=dataset,
                cases=cases,
                experiment=experiment,
            )
            _write_jsonl(run_dir / "shared_pipeline.raw.jsonl", shared_pipeline)
            for group_name, results in group_results.items():
                _write_jsonl(run_dir / f"{group_name}.raw.jsonl", results)
        elif experiment.target == "evidence_orchestration":
            group_results, shared_pipeline = await _run_evidence_cases_bounded(
                client=client,
                dataset=dataset,
                cases=cases,
                experiment=experiment,
            )
            _write_jsonl(run_dir / "shared_pipeline.raw.jsonl", shared_pipeline)
            for group_name, results in group_results.items():
                _write_jsonl(run_dir / f"{group_name}.raw.jsonl", results)
        elif _is_shared_rerank_experiment(experiment):
            shared_candidates = await _retrieve_shared_candidates_bounded(
                client=client,
                dataset=dataset,
                cases=cases,
                experiment=experiment,
            )
            _write_jsonl(run_dir / "shared_candidates.raw.jsonl", shared_candidates)
            group_results = {}
            for group_name, group in (
                ("baseline", experiment.baseline),
                ("candidate", experiment.candidate),
            ):
                provider = _build_eval_rerank_provider(group.rerank_provider)
                results = await _run_shared_rerank_cases_bounded(
                    shared_rows=shared_candidates,
                    cases=cases,
                    group_name=group_name,
                    group=group,
                    provider=provider,
                    concurrency=experiment.concurrency,
                )
                group_results[group_name] = results
                _write_jsonl(run_dir / f"{group_name}.raw.jsonl", results)
        else:
            group_results = {}
            groups: tuple[tuple[str, ExperimentGroup], ...]
            if experiment.baseline_run_dir:
                frozen_rows = _load_frozen_baseline_rows(
                    experiment=experiment,
                    cases=cases,
                )
                group_results["baseline"] = frozen_rows
                _write_jsonl(run_dir / "baseline.raw.jsonl", frozen_rows)
                groups = (("candidate", experiment.candidate),)
            else:
                groups = (
                    ("baseline", experiment.baseline),
                    ("candidate", experiment.candidate),
                )
            for group_name, group in groups:
                warmup_count = min(experiment.warmup_cases, len(cases))
                for case in cases[:warmup_count]:
                    await _run_case(
                        client=client,
                        dataset=dataset,
                        case=case,
                        group_name=group_name,
                        group=group,
                        keep_result=False,
                    )
                results = await _run_cases_bounded(
                    client=client,
                    dataset=dataset,
                    cases=cases,
                    group_name=group_name,
                    group=group,
                    concurrency=experiment.concurrency,
                )
                group_results[group_name] = results
                _write_jsonl(run_dir / f"{group_name}.raw.jsonl", results)

    finished_at = datetime.now(UTC)
    manifest["finished_at"] = finished_at.isoformat()
    manifest["duration_seconds"] = round((finished_at - started_at).total_seconds(), 3)
    manifest["groups"] = {
        name: {
            "total_cases": len(results),
            "successful_cases": sum(item["error"] is None for item in results),
            "failed_cases": sum(item["error"] is not None for item in results),
        }
        for name, results in group_results.items()
    }
    manifest["status"] = (
        "retrieval_complete"
        if all(item["error"] is None for results in group_results.values() for item in results)
        else "retrieval_complete_with_failures"
    )
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _load_frozen_baseline_rows(
    *,
    experiment: ExperimentConfig,
    cases: list[EvalCase],
) -> list[dict[str, Any]]:
    assert experiment.baseline_run_dir is not None
    project_root = Path(__file__).resolve().parents[2]
    source_dir = Path(experiment.baseline_run_dir)
    if not source_dir.is_absolute():
        source_dir = project_root / source_dir
    source_path = source_dir / f"{experiment.baseline_group}.raw.jsonl"
    if not source_path.is_file():
        raise EvaluationRunError(f"冻结 baseline raw 不存在: {source_path}")
    rows_by_case = {
        str(row.get("case_id")): row
        for row in _read_jsonl(source_path)
    }
    frozen_rows: list[dict[str, Any]] = []
    for case in cases:
        row = rows_by_case.get(case.id)
        if row is None:
            raise EvaluationRunError(f"冻结 baseline 缺少题目: {case.id}")
        frozen_rows.append({**row, "group": "baseline"})
    return frozen_rows


async def retry_failed_shared_rerank(
    *,
    run_dir: Path,
    group_name: str,
    concurrency: int = 1,
) -> int:
    """Retry only failed rerank rows while reusing the persisted shared candidates."""
    if group_name not in {"baseline", "candidate"}:
        raise EvaluationRunError("group_name 必须是 baseline 或 candidate")
    experiment = ExperimentConfig.model_validate(
        json.loads((run_dir / "experiment.snapshot.json").read_text(encoding="utf-8"))
    )
    dataset = EvalDataset.model_validate(
        json.loads((run_dir / "dataset.snapshot.json").read_text(encoding="utf-8"))
    )
    if not _is_shared_rerank_experiment(experiment):
        raise EvaluationRunError("当前运行目录不是共同候选精排实验")

    shared_rows = _read_jsonl(run_dir / "shared_candidates.raw.jsonl")
    result_path = run_dir / f"{group_name}.raw.jsonl"
    result_rows = _read_jsonl(result_path)
    failed_case_ids = {
        str(row.get("case_id")) for row in result_rows if row.get("error") is not None
    }
    if not failed_case_ids:
        return 0

    cases_by_id = {case.id: case for case in dataset.cases_for_suite(experiment.suite)}
    shared_by_id = {str(row.get("case_id")): row for row in shared_rows}
    retry_case_ids = [
        str(row.get("case_id"))
        for row in result_rows
        if str(row.get("case_id")) in failed_case_ids
    ]
    try:
        retry_cases = [cases_by_id[case_id] for case_id in retry_case_ids]
        retry_shared = [shared_by_id[case_id] for case_id in retry_case_ids]
    except KeyError as exc:
        raise EvaluationRunError(f"失败题缺少数据或共同候选: {exc.args[0]}") from exc

    group = getattr(experiment, group_name)
    provider = _build_eval_rerank_provider(group.rerank_provider)
    retried_rows = await _run_shared_rerank_cases_bounded(
        shared_rows=retry_shared,
        cases=retry_cases,
        group_name=group_name,
        group=group,
        provider=provider,
        concurrency=concurrency,
    )
    replacements = {str(row["case_id"]): row for row in retried_rows}
    merged_rows = [replacements.get(str(row.get("case_id")), row) for row in result_rows]
    _write_jsonl(result_path, merged_rows)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for name in ("baseline", "candidate"):
        rows = _read_jsonl(run_dir / f"{name}.raw.jsonl")
        all_rows.extend(rows)
        groups[name] = {
            "total_cases": len(rows),
            "successful_cases": sum(row.get("error") is None for row in rows),
            "failed_cases": sum(row.get("error") is not None for row in rows),
        }
    manifest["groups"] = groups
    manifest["status"] = (
        "retrieval_complete"
        if all(row.get("error") is None for row in all_rows)
        else "retrieval_complete_with_failures"
    )
    _write_json(manifest_path, manifest)
    return len(retried_rows)


def _is_shared_rerank_experiment(experiment: ExperimentConfig) -> bool:
    return (
        experiment.target not in {
            "evidence_orchestration",
            "research_effect_v2",
            "research_supplement_v2",
        }
        and
        experiment.baseline.rerank_provider is not None
        and experiment.candidate.rerank_provider is not None
    )


async def _run_research_cases_bounded(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    cases: list[EvalCase],
    experiment: ExperimentConfig,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(experiment.concurrency)

    async def worker(case: EvalCase) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        async with semaphore:
            if experiment.target == "research_effect_v2":
                # _run_evidence_case returns raw Top10 first and EvidencePack second.
                # The latter is the declared Single Retrieve baseline here.
                _, baseline, shared = await _run_evidence_case(
                    client=client,
                    dataset=dataset,
                    case=case,
                    experiment=experiment.model_copy(
                        deep=True,
                        update={"candidate": experiment.baseline},
                    ),
                )
                baseline["group"] = "baseline"
                candidate = await _run_research_http_case(
                    client=client,
                    dataset=dataset,
                    case=case,
                    group_name="candidate",
                    group=experiment.candidate,
                    allow_supplement=True,
                )
                return baseline, candidate, shared

            baseline = await _run_research_http_case(
                client=client,
                dataset=dataset,
                case=case,
                group_name="baseline",
                group=experiment.baseline,
                allow_supplement=False,
            )
            candidate = await _run_research_http_case(
                client=client,
                dataset=dataset,
                case=case,
                group_name="candidate",
                group=experiment.candidate,
                allow_supplement=True,
            )
            baseline_first = (baseline.get("diagnostics") or {}).get("first_round")
            candidate_first = (candidate.get("diagnostics") or {}).get("first_round")
            shared = {
                "case_id": case.id,
                "baseline_first_round": baseline_first,
                "candidate_first_round": candidate_first,
                "shared": _canonical_first_round(baseline_first)
                == _canonical_first_round(candidate_first),
                "error": baseline.get("error") or candidate.get("error"),
            }
            return baseline, candidate, shared

    triples = list(await asyncio.gather(*(worker(case) for case in cases)))
    return (
        {
            "baseline": [item[0] for item in triples],
            "candidate": [item[1] for item in triples],
        },
        [item[2] for item in triples],
    )


def _canonical_first_round(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "purpose": value.get("purpose"),
        "candidate_chunk_ids": value.get("candidate_chunk_ids") or [],
        "tasks": [
            {
                "query_id": item.get("query_id"),
                "query": item.get("query"),
                "aspect_ids": item.get("aspect_ids") or [],
                "success": item.get("success"),
                "chunk_ids": item.get("chunk_ids") or [],
            }
            for item in value.get("tasks") or []
            if isinstance(item, dict)
        ],
    }


async def _run_research_http_case(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    case: EvalCase,
    group_name: str,
    group: ExperimentGroup,
    allow_supplement: bool,
) -> dict[str, Any]:
    kb_aliases = case.resolved_kb_aliases()
    kb_ids = [dataset.knowledge_bases[alias].kb_id for alias in kb_aliases]
    payload = _build_research_payload(case=case, kb_ids=kb_ids, group=group)
    started_at = perf_counter()
    try:
        response = await client.post(
            "/api/v1/rag/research",
            json=payload,
            headers={"X-Eval-Allow-Supplement": "true" if allow_supplement else "false"},
        )
        latency_ms = (perf_counter() - started_at) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise EvaluationRunError(
                f"research error code={body.get('code')}: {body.get('msg') or 'unknown error'}"
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise EvaluationRunError("research 响应缺少 data")
        if "chunks" in data or "retrieved_chunks" in data:
            raise EvaluationRunError("Research 响应不应包含 chunks")
        evidence_pack = data.get("evidence_pack")
        if not isinstance(evidence_pack, dict):
            raise EvaluationRunError("Research 响应缺少 evidence_pack")
        expected_missing = case.expected_status == "missing"
        contexts, evidence_diagnostics = _contexts_from_evidence_pack(
            evidence_pack=evidence_pack,
            kb_ids=kb_ids,
            index_version=group.index_version,
            max_items=int(group.evidence_options.get("max_items") or 10),
            allow_empty=expected_missing,
        )
        metadata = data.get("metadata") or {}
        rounds = data.get("rounds") or []
        answer = data.get("answer")
        if not expected_missing and not str(answer or "").strip():
            raise EvaluationRunError("正常 Research 响应的 answer 为空")
        _assert_research_limits(metadata=metadata, rounds=rounds, allow_supplement=allow_supplement)
        citations_valid = _answer_citations_valid(answer, evidence_pack)
        if not citations_valid:
            raise EvaluationRunError("answer 引用无法映射到 EvidencePack")
        diagnostics = {
            **evidence_diagnostics,
            "answer_present": bool(str(answer or "").strip()),
            "answer_citations_valid": citations_valid,
            "round_count": int(metadata.get("round_count") or 0),
            "retrieval_task_count": int(metadata.get("retrieval_task_count") or 0),
            "research_llm_call_count": int(metadata.get("llm_call_count") or 0),
            "input_tokens": int(metadata.get("input_tokens") or 0),
            "output_tokens": int(metadata.get("output_tokens") or 0),
            "initial_evidence_status": metadata.get("initial_evidence_status"),
            "stop_reason": metadata.get("stop_reason"),
            "degraded": bool(metadata.get("degraded")),
            "supplement_executed": int(metadata.get("round_count") or 0) > 1,
            "first_round": rounds[0] if rounds else None,
        }
        row = {
            **_case_result_fields(case, kb_aliases),
            "group": group_name,
            "answer": answer,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 3),
            "stage_latency_ms": {
                item.get("stage"): item.get("latency_ms")
                for item in metadata.get("stages") or []
                if isinstance(item, dict)
            },
            "application_model_calls": int(metadata.get("llm_call_count") or 0)
            + int(metadata.get("retrieval_task_count") or 0),
            "retry_count": 0,
            "effective_config": {
                "execution_mode": group.execution_mode,
                "index_version": group.index_version,
                "top_k": group.top_k,
                "rerank_provider": "qwen3",
                "evidence_enabled": True,
            },
            "diagnostics": diagnostics,
            "error": None,
        }
        if expected_missing and not contexts:
            row["skip_ragas_reason"] = "expected_missing_no_context"
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            **_case_result_fields(case, kb_aliases),
            "group": group_name,
            "answer": None,
            "contexts": [],
            "latency_ms": round((perf_counter() - started_at) * 1000, 3),
            "stage_latency_ms": {},
            "application_model_calls": None,
            "retry_count": 0,
            "effective_config": {},
            "diagnostics": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _build_research_payload(
    *, case: EvalCase, kb_ids: list[str], group: ExperimentGroup
) -> dict[str, Any]:
    payload = {
        "user_id": EVAL_USER_ID,
        "query": case.question,
    }
    payload["kb_id" if len(kb_ids) == 1 else "kb_ids"] = kb_ids[0] if len(kb_ids) == 1 else kb_ids
    return payload


def _assert_research_limits(
    *, metadata: dict[str, Any], rounds: list[dict[str, Any]], allow_supplement: bool
) -> None:
    if int(metadata.get("round_count") or 0) > (2 if allow_supplement else 1):
        raise EvaluationRunError("Research 检索轮数超过硬上限")
    if int(metadata.get("llm_call_count") or 0) > 3:
        raise EvaluationRunError("Research LLM 调用超过硬上限")
    if len(rounds) > (2 if allow_supplement else 1):
        raise EvaluationRunError("Research rounds 超过硬上限")
    for index, round_item in enumerate(rounds):
        limit = 3 if index == 0 else 2
        if len(round_item.get("tasks") or []) > limit:
            raise EvaluationRunError("Research 子查询数超过硬上限")


def _answer_citations_valid(answer: Any, evidence_pack: dict[str, Any]) -> bool:
    import re

    evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in evidence_pack.get("items") or []
        if isinstance(item, dict)
    }
    references = re.findall(r"\[(E\d+)\]", str(answer or ""))
    return set(references) <= evidence_ids


async def _run_evidence_cases_bounded(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    cases: list[EvalCase],
    experiment: ExperimentConfig,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(experiment.concurrency)

    async def worker(case: EvalCase) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        async with semaphore:
            return await _run_evidence_case(
                client=client,
                dataset=dataset,
                case=case,
                experiment=experiment,
            )

    rows = list(await asyncio.gather(*(worker(case) for case in cases)))
    return (
        {
            "baseline": [row[0] for row in rows],
            "candidate": [row[1] for row in rows],
        },
        [row[2] for row in rows],
    )


async def _run_evidence_case(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    case: EvalCase,
    experiment: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    kb_aliases = case.resolved_kb_aliases()
    kb_ids = [dataset.knowledge_bases[alias].kb_id for alias in kb_aliases]
    group = experiment.candidate
    payload = _build_payload(case=case, kb_ids=kb_ids, group=group)
    started_at = perf_counter()
    try:
        response = await client.post("/api/v1/rag/retrieve", json=payload)
        full_latency_ms = (perf_counter() - started_at) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise EvaluationRunError(
                f"retrieve error code={body.get('code')}: {body.get('msg') or 'unknown error'}"
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise EvaluationRunError("retrieve 响应缺少 data")

        baseline_contexts = _extract_contexts(data)
        if len(baseline_contexts) != group.top_k:
            raise EvaluationRunError(
                f"证据编排实验必须返回 Top{group.top_k}: actual={len(baseline_contexts)}"
            )
        _assert_index_version(baseline_contexts, group.index_version)
        effective_config = _extract_effective_config(data, group)
        _assert_effective_config(group, effective_config)
        rerank_metadata = (data.get("metadata") or {}).get("rerank") or {}
        if rerank_metadata.get("degraded"):
            raise EvaluationRunError(
                f"Qwen3 精排发生降级: {rerank_metadata.get('error') or 'unknown error'}"
            )
        if effective_config.get("rerank_provider") != "qwen3":
            raise EvaluationRunError(
                "证据编排实验未使用 Qwen3 精排: "
                f"actual={effective_config.get('rerank_provider')}"
            )

        evidence_pack = data.get("evidence_pack")
        metadata = data.get("metadata") or {}
        evidence_metadata = metadata.get("evidence") or {}
        if evidence_metadata.get("degraded"):
            raise EvaluationRunError(
                f"证据编排发生降级: {evidence_metadata.get('error') or 'unknown error'}"
            )
        if not evidence_metadata.get("executed") or not isinstance(evidence_pack, dict):
            raise EvaluationRunError("证据编排未执行或响应缺少 evidence_pack")
        evidence_contexts, evidence_diagnostics = _contexts_from_evidence_pack(
            evidence_pack=evidence_pack,
            kb_ids=kb_ids,
            index_version=group.index_version,
            max_items=int(group.evidence_options.get("max_items") or 10),
            allow_empty=case.expected_status == "missing",
        )
        evidence_latency_ms = float(evidence_metadata.get("latency_ms") or 0)
        baseline_latency_ms = max(0.0, full_latency_ms - evidence_latency_ms)
        common = _case_result_fields(case, kb_aliases)
        stage_latency = {
            "query_processing": (metadata.get("query_processing") or {}).get("latency_ms", 0),
            "retrieval": metadata.get("latency_ms"),
            "rerank": (metadata.get("rerank") or {}).get("latency_ms", 0),
            "evidence": evidence_latency_ms,
        }
        baseline = {
            **common,
            "group": "baseline",
            "answer": None,
            "contexts": baseline_contexts,
            "latency_ms": round(baseline_latency_ms, 3),
            "stage_latency_ms": {**stage_latency, "evidence": 0},
            "application_model_calls": 1,
            "retry_count": 0,
            "effective_config": {**effective_config, "evidence_enabled": False},
            "diagnostics": {
                "evidence_count": len(baseline_contexts),
                "evidence_chars": sum(len(item["text"]) for item in baseline_contexts),
            },
            "error": None,
        }
        candidate = {
            **common,
            "group": "candidate",
            "answer": None,
            "contexts": evidence_contexts,
            "latency_ms": round(full_latency_ms, 3),
            "stage_latency_ms": stage_latency,
            "application_model_calls": 2,
            "retry_count": 0,
            "effective_config": effective_config,
            "diagnostics": evidence_diagnostics,
            "error": None,
        }
        if case.expected_status == "missing" and not baseline_contexts:
            baseline["skip_ragas_reason"] = "expected_missing_no_context"
        if case.expected_status == "missing" and not evidence_contexts:
            candidate["skip_ragas_reason"] = "expected_missing_no_context"
        shared = {
            "case_id": case.id,
            "retrieved_chunk_ids": [item["chunk_id"] for item in baseline_contexts],
            "evidence_chunk_ids": [item["chunk_id"] for item in evidence_contexts],
            "rerank_provider": effective_config.get("rerank_provider"),
            "index_version": group.index_version,
            "full_latency_ms": round(full_latency_ms, 3),
            "evidence_latency_ms": round(evidence_latency_ms, 3),
            "error": None,
        }
        return baseline, candidate, shared
    except Exception as exc:  # noqa: BLE001 - each case failure is persisted independently
        latency_ms = (perf_counter() - started_at) * 1000
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_rows = []
        for group_name in ("baseline", "candidate"):
            failed_rows.append(
                {
                    **_case_result_fields(case, kb_aliases),
                    "group": group_name,
                    "answer": None,
                    "contexts": [],
                    "latency_ms": round(latency_ms, 3),
                    "stage_latency_ms": {},
                    "application_model_calls": None,
                    "retry_count": 0,
                    "effective_config": {},
                    "diagnostics": {},
                    "error": error,
                }
            )
        return failed_rows[0], failed_rows[1], {"case_id": case.id, "error": error}


def _contexts_from_evidence_pack(
    *,
    evidence_pack: dict[str, Any],
    kb_ids: list[str],
    index_version: str,
    max_items: int,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = evidence_pack.get("items")
    if not isinstance(items, list) or (not items and not allow_empty):
        raise EvaluationRunError("evidence_pack.items 为空")
    if len(items) > max_items:
        raise EvaluationRunError(
            f"证据数量超过上限: max={max_items}, actual={len(items)}"
        )
    chunk_ids: list[str] = []
    contexts: list[dict[str, Any]] = []
    traceable = 0
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise EvaluationRunError("EvidenceItem 格式无效")
        chunk_id = str(item.get("chunk_id") or "")
        content = str(item.get("content") or "").strip()
        kb_id = str(item.get("kb_id") or "")
        item_version = str(item.get("index_version") or "")
        if not chunk_id or not content:
            raise EvaluationRunError("EvidenceItem 缺少 chunk_id 或原始正文")
        if kb_id not in kb_ids or item_version != index_version:
            raise EvaluationRunError("EvidenceItem 超出知识库范围或索引版本不一致")
        chunk_ids.append(chunk_id)
        traceable += 1
        contexts.append(
            {
                "rank": rank,
                "text": content,
                "anchor_text": content,
                "retrieval_text": None,
                "document_id": str(item.get("document_id") or ""),
                "chunk_id": chunk_id,
                "title": str(item.get("title") or ""),
                "heading": item.get("heading_path"),
                "stable_document_id": _normalize_document_name(str(item.get("title") or "")),
                "score": None,
                "vector_score": None,
                "bm25_score": None,
                "rerank_score": None,
                "retrieval_source": item.get("source"),
                "index_version": item_version,
            }
        )
    if len(chunk_ids) != len(set(chunk_ids)):
        raise EvaluationRunError("evidence_pack.items 包含重复 chunk_id")
    return contexts, {
        "evidence_count": len(items),
        "evidence_chars": sum(len(item["text"]) for item in contexts),
        "evidence_status": evidence_pack.get("status"),
        "missing_aspects": evidence_pack.get("missing_aspects") or [],
        "citation_traceable_items": traceable,
        "citation_total_items": len(items),
        "citation_traceability_rate": traceable / len(items) if items else 1.0,
    }


def _build_eval_rerank_provider(provider_name: str | None) -> Any:
    if provider_name == "noop":
        from app.providers.rerank.noop import NoopRerankProvider

        return NoopRerankProvider()
    if provider_name == "llm":
        from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
        from app.providers.rerank.llm import LLMRerankProvider

        return LLMRerankProvider(llm_provider=OpenAICompatibleLLMProvider())
    if provider_name == "qwen3":
        from app.providers.rerank.qwen3 import Qwen3RerankProvider

        return Qwen3RerankProvider()
    raise EvaluationRunError(f"不支持的离线精排 Provider: {provider_name}")


async def _retrieve_shared_candidates_bounded(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    cases: list[EvalCase],
    experiment: ExperimentConfig,
) -> list[dict[str, Any]]:
    candidate_top_k = experiment.shared_candidate_top_k
    if candidate_top_k is None:
        raise EvaluationRunError("精排实验缺少 shared_candidate_top_k")
    retrieval_group = experiment.baseline.model_copy(
        deep=True,
        update={
            "top_k": candidate_top_k,
            "rerank_options": {
                **experiment.baseline.rerank_options,
                "enabled": False,
            },
            "rerank_provider": None,
        },
    )
    semaphore = asyncio.Semaphore(experiment.concurrency)

    async def worker(case: EvalCase) -> dict[str, Any]:
        async with semaphore:
            return await _retrieve_shared_candidate_case(
                client=client,
                dataset=dataset,
                case=case,
                group=retrieval_group,
            )

    return list(await asyncio.gather(*(worker(case) for case in cases)))


async def _retrieve_shared_candidate_case(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    case: EvalCase,
    group: ExperimentGroup,
) -> dict[str, Any]:
    kb_aliases = case.resolved_kb_aliases()
    kb_ids = [dataset.knowledge_bases[alias].kb_id for alias in kb_aliases]
    payload = _build_payload(case=case, kb_ids=kb_ids, group=group)
    started_at = perf_counter()
    try:
        response = await client.post("/api/v1/rag/retrieve", json=payload)
        latency_ms = (perf_counter() - started_at) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise EvaluationRunError(
                f"retrieve error code={body.get('code')}: {body.get('msg') or 'unknown error'}"
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise EvaluationRunError("retrieve 响应缺少 data")
        candidates = _extract_rerank_candidates(data)
        contexts = _contexts_from_chunks(candidates)
        _assert_index_version(contexts, group.index_version)
        effective_config = _extract_effective_config(data, group)
        _assert_effective_config(group, effective_config)
        metadata = data.get("metadata") or {}
        return {
            **_case_result_fields(case, kb_aliases),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "latency_ms": round(latency_ms, 3),
            "stage_latency_ms": {
                "retrieval": metadata.get("latency_ms"),
            },
            "effective_config": effective_config,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - each case failure is persisted independently
        latency_ms = (perf_counter() - started_at) * 1000
        return {
            **_case_result_fields(case, kb_aliases),
            "candidates": [],
            "candidate_count": 0,
            "latency_ms": round(latency_ms, 3),
            "stage_latency_ms": {},
            "effective_config": {},
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


async def _run_shared_rerank_cases_bounded(
    *,
    shared_rows: list[dict[str, Any]],
    cases: list[EvalCase],
    group_name: str,
    group: ExperimentGroup,
    provider: Any,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(shared: dict[str, Any], case: EvalCase) -> dict[str, Any]:
        async with semaphore:
            return await _run_shared_rerank_case(
                shared=shared,
                case=case,
                group_name=group_name,
                group=group,
                provider=provider,
            )

    return list(
        await asyncio.gather(
            *(worker(shared, case) for shared, case in zip(shared_rows, cases, strict=True))
        )
    )


async def _run_shared_rerank_case(
    *,
    shared: dict[str, Any],
    case: EvalCase,
    group_name: str,
    group: ExperimentGroup,
    provider: Any,
) -> dict[str, Any]:
    kb_aliases = case.resolved_kb_aliases()
    shared_latency_ms = float(shared.get("latency_ms") or 0)
    if shared.get("error") is not None:
        return {
            **_case_result_fields(case, kb_aliases),
            "group": group_name,
            "answer": None,
            "contexts": [],
            "latency_ms": round(shared_latency_ms, 3),
            "stage_latency_ms": shared.get("stage_latency_ms") or {},
            "application_model_calls": None,
            "retry_count": 0,
            "effective_config": {},
            "error": shared["error"],
        }

    rerank_started_at = perf_counter()
    try:
        candidates = [dict(item) for item in shared.get("candidates") or []]
        reranked = await provider.rerank(
            query=case.question,
            chunks=candidates,
            top_n=group.top_k,
        )
        rerank_latency_ms = (perf_counter() - rerank_started_at) * 1000
        _assert_rerank_output(
            candidates=candidates,
            reranked=reranked,
            top_n=group.top_k,
        )
        contexts = _contexts_from_chunks(reranked)
        _assert_index_version(contexts, group.index_version)
        shared_effective_config = shared.get("effective_config") or {}
        rewrite_enabled = bool(shared_effective_config.get("rewrite_enabled"))
        application_model_calls = int(rewrite_enabled) + (
            0 if group.rerank_provider == "noop" else 1
        )
        effective_config = {
            "profile": group.profile,
            "retrieval_mode": group.retrieval_options.get("mode"),
            "rewrite_enabled": rewrite_enabled,
            "synonym_enabled": False,
            "rerank_enabled": group.rerank_provider != "noop",
            "rerank_provider": group.rerank_provider,
            "evidence_enabled": False,
            "shared_candidate_top_k": len(candidates),
            "top_k": group.top_k,
            "index_version": group.index_version,
        }
        return {
            **_case_result_fields(case, kb_aliases),
            "group": group_name,
            "answer": None,
            "contexts": contexts,
            "latency_ms": round(shared_latency_ms + rerank_latency_ms, 3),
            "stage_latency_ms": {
                "query_processing": 0,
                "retrieval": (shared.get("stage_latency_ms") or {}).get("retrieval"),
                "rerank": round(rerank_latency_ms, 3),
                "evidence": 0,
            },
            "application_model_calls": application_model_calls,
            "retry_count": 0,
            "effective_config": effective_config,
            "ranking_qrels": _ranking_qrels(case=case, candidates=candidates),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - each case failure is persisted independently
        rerank_latency_ms = (perf_counter() - rerank_started_at) * 1000
        return {
            **_case_result_fields(case, kb_aliases),
            "group": group_name,
            "answer": None,
            "contexts": [],
            "latency_ms": round(shared_latency_ms + rerank_latency_ms, 3),
            "stage_latency_ms": {
                "retrieval": (shared.get("stage_latency_ms") or {}).get("retrieval"),
                "rerank": round(rerank_latency_ms, 3),
            },
            "application_model_calls": (
                int(bool((shared.get("effective_config") or {}).get("rewrite_enabled")))
                + (0 if group.rerank_provider == "noop" else 1)
            ),
            "retry_count": 0,
            "effective_config": {},
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _assert_rerank_output(
    *,
    candidates: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    top_n: int,
) -> None:
    expected_count = min(top_n, len(candidates))
    if len(reranked) != expected_count:
        raise EvaluationRunError(
            f"精排返回数量不正确: expected={expected_count}, actual={len(reranked)}"
        )
    candidate_ids = {str(item.get("chunk_id") or "") for item in candidates}
    result_ids = [str(item.get("chunk_id") or "") for item in reranked]
    if len(result_ids) != len(set(result_ids)):
        raise EvaluationRunError("精排结果包含重复 chunk_id")
    unknown_ids = sorted(set(result_ids) - candidate_ids)
    if unknown_ids:
        raise EvaluationRunError("精排结果包含候选池外 chunk: " + ", ".join(unknown_ids))


def _ranking_qrels(
    *,
    case: EvalCase,
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    expected_documents = {
        _normalize_document_name(item).casefold()
        for item in case.expected_documents
    }
    expected_headings = [
        "".join(str(item).casefold().split())
        for item in case.expected_headings
        if str(item).strip()
    ]
    labels: dict[str, int] = {}
    for candidate in candidates:
        chunk_id = str(candidate.get("chunk_id") or "")
        if not chunk_id:
            continue
        explicit = case.relevance.get(chunk_id)
        if explicit is not None:
            labels[chunk_id] = int(explicit)
            continue
        title = _normalize_document_name(str(candidate.get("title") or "")).casefold()
        if title not in expected_documents:
            labels[chunk_id] = 0
            continue
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        heading = (
            metadata.get("heading_path")
            or metadata.get("section_title")
            or metadata.get("heading")
            or ""
        )
        normalized_heading = "".join(str(heading).casefold().split())
        labels[chunk_id] = (
            3
            if expected_headings
            and any(item in normalized_heading for item in expected_headings)
            else 2
        )
    return labels


async def _run_cases_bounded(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    cases: list[EvalCase],
    group_name: str,
    group: ExperimentGroup,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(case: EvalCase) -> dict[str, Any]:
        async with semaphore:
            return await _run_case(
                client=client,
                dataset=dataset,
                case=case,
                group_name=group_name,
                group=group,
                keep_result=True,
            )

    return list(await asyncio.gather(*(worker(case) for case in cases)))


async def _run_case(
    *,
    client: httpx.AsyncClient,
    dataset: EvalDataset,
    case: EvalCase,
    group_name: str,
    group: ExperimentGroup,
    keep_result: bool,
) -> dict[str, Any]:
    kb_aliases = case.resolved_kb_aliases()
    kb_ids = [dataset.knowledge_bases[alias].kb_id for alias in kb_aliases]
    payload = _build_payload(case=case, kb_ids=kb_ids, group=group)
    started_at = perf_counter()
    try:
        response = await client.post("/api/v1/rag/retrieve", json=payload)
        latency_ms = (perf_counter() - started_at) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise EvaluationRunError(
                f"retrieve error code={body.get('code')}: {body.get('msg') or 'unknown error'}"
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise EvaluationRunError("retrieve 响应缺少 data")
        contexts = _extract_contexts(data)
        _assert_index_version(contexts, group.index_version)
        effective_config = _extract_effective_config(data, group)
        _assert_effective_config(group, effective_config)
        if not keep_result:
            return {}
        return _success_result(
            case=case,
            group_name=group_name,
            kb_aliases=kb_aliases,
            data=data,
            contexts=contexts,
            effective_config=effective_config,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - each case failure is persisted independently
        latency_ms = (perf_counter() - started_at) * 1000
        if not keep_result:
            return {}
        return {
            "case_id": case.id,
            "group": group_name,
            "kb_aliases": kb_aliases,
            "question": case.question,
            "ground_truth": case.ground_truth,
            "case_type": case.case_type,
            "tags": case.tags,
            "expected_documents": case.expected_documents,
            "expected_headings": case.expected_headings,
            "answer": None,
            "contexts": [],
            "latency_ms": round(latency_ms, 3),
            "stage_latency_ms": {},
            "application_model_calls": None,
            "retry_count": 0,
            "effective_config": {},
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _build_payload(
    *,
    case: EvalCase,
    kb_ids: list[str],
    group: ExperimentGroup,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": EVAL_USER_ID,
        "query": case.question,
        "profile": group.profile,
        "index_version": group.index_version,
        "observability_enabled": False,
    }
    if group.profile == "custom":
        payload.update(
            {
                "top_k": group.top_k,
                "retrieval_options": group.retrieval_options,
                "rerank_options": group.rerank_options,
                "evidence_options": group.evidence_options,
                "query_options": group.query_options,
            }
        )
    if len(kb_ids) == 1:
        payload["kb_id"] = kb_ids[0]
    else:
        payload["kb_ids"] = kb_ids
    return payload


def _extract_contexts(data: dict[str, Any]) -> list[dict[str, Any]]:
    return _contexts_from_chunks(data.get("retrieved_chunks") or [])


def _extract_rerank_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for chunk in data.get("retrieved_chunks") or []:
        if not isinstance(chunk, dict):
            continue
        candidates.append(
            {
                **chunk,
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "document_id": str(chunk.get("document_id") or ""),
                "title": str(chunk.get("title") or ""),
                "content": str(chunk.get("content") or ""),
                "score": _optional_float(chunk.get("score")) or 0.0,
            }
        )
    return candidates


def _contexts_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        heading = (
            metadata.get("heading_path")
            or metadata.get("section_title")
            or metadata.get("heading")
        )
        title = str(chunk.get("title") or "")
        expanded_context = chunk.get("context")
        context_text = (
            str(expanded_context.get("content") or "")
            if isinstance(expanded_context, dict)
            else ""
        )
        contexts.append(
            {
                "rank": rank,
                "text": context_text or str(chunk.get("content") or ""),
                "anchor_text": str(chunk.get("content") or ""),
                "retrieval_text": chunk.get("retrieval_text"),
                "document_id": str(chunk.get("document_id") or ""),
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "title": title,
                "heading": str(heading) if heading else None,
                "stable_document_id": _normalize_document_name(title),
                "score": _optional_float(chunk.get("score")),
                "vector_score": _optional_float(chunk.get("vector_score")),
                "bm25_score": _optional_float(chunk.get("bm25_score")),
                "rerank_score": _optional_float(chunk.get("rerank_score")),
                "retrieval_source": chunk.get("retrieval_source"),
                "index_version": chunk.get("index_version"),
            }
        )
    return contexts


def _case_result_fields(case: EvalCase, kb_aliases: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "kb_aliases": kb_aliases,
        "question": case.question,
        "ground_truth": case.ground_truth,
        "case_type": case.case_type,
        "tags": case.tags,
        "expected_documents": case.expected_documents,
        "expected_headings": case.expected_headings,
        "required_aspects": [item.model_dump(mode="json") for item in case.required_aspects],
        "expected_status": case.expected_status,
        "requires_multi_hop": case.requires_multi_hop,
        "dependency": case.dependency.model_dump(mode="json") if case.dependency else None,
    }


def _assert_index_version(contexts: list[dict[str, Any]], expected_version: str) -> None:
    unexpected = sorted(
        {
            str(context.get("index_version"))
            for context in contexts
            if context.get("index_version") != expected_version
        }
    )
    if unexpected:
        raise EvaluationRunError(
            f"评测请求返回了错误索引版本: expected={expected_version}, "
            f"actual={', '.join(unexpected)}"
        )


def _assert_legacy_index(contexts: list[dict[str, Any]]) -> None:
    _assert_index_version(contexts, LEGACY_INDEX_VERSION)


def _extract_effective_config(
    data: dict[str, Any],
    group: ExperimentGroup,
) -> dict[str, Any]:
    metadata = data.get("metadata") or {}
    policy = metadata.get("tenant_policy") or {}
    query_processing = metadata.get("query_processing") or {}
    rerank = metadata.get("rerank") or {}
    evidence = metadata.get("evidence") or {}
    retrieval = metadata.get("retrieval") or {}
    requested_synonym = group.query_options.get("synonym_enabled")
    return {
        "profile": policy.get("retrieve_profile"),
        "retrieval_mode": policy.get("effective_mode") or retrieval.get("mode"),
        "rewrite_enabled": bool(policy.get("effective_query_rewrite")),
        "synonym_enabled": query_processing.get(
            "synonym_enabled",
            False if requested_synonym is None else bool(requested_synonym),
        ),
        "rerank_enabled": bool(rerank.get("enabled")),
        "evidence_enabled": bool(evidence.get("enabled")),
        "rerank_provider": rerank.get("provider"),
        "top_k": metadata.get("top_k"),
        "index_version": group.index_version,
    }


def _assert_effective_config(
    group: ExperimentGroup,
    effective: dict[str, Any],
) -> None:
    expected_mode = group.retrieval_options.get("mode")
    if expected_mode is not None and effective["retrieval_mode"] != expected_mode:
        raise EvaluationRunError(
            f"检索模式未生效: expected={expected_mode}, actual={effective['retrieval_mode']}"
        )
    expected_rewrite = group.query_options.get("enabled")
    if expected_rewrite is not None and effective["rewrite_enabled"] is not bool(expected_rewrite):
        raise EvaluationRunError(
            "Query Rewrite 开关未按实验配置生效: "
            f"expected={bool(expected_rewrite)}, actual={effective['rewrite_enabled']}"
        )
    expected_synonym = group.query_options.get("synonym_enabled")
    if expected_synonym is not None and effective["synonym_enabled"] is not bool(expected_synonym):
        raise EvaluationRunError(
            "同义词开关未按实验配置生效: "
            f"expected={bool(expected_synonym)}, actual={effective['synonym_enabled']}"
        )
    expected_rerank = group.rerank_options.get("enabled")
    if expected_rerank is not None and effective["rerank_enabled"] is not bool(expected_rerank):
        raise EvaluationRunError(
            "重排开关未按实验配置生效: "
            f"expected={bool(expected_rerank)}, actual={effective['rerank_enabled']}"
        )
    expected_evidence = group.evidence_options.get("enabled")
    if expected_evidence is not None and effective["evidence_enabled"] is not bool(expected_evidence):
        raise EvaluationRunError(
            "证据编排开关未按实验配置生效: "
            f"expected={bool(expected_evidence)}, actual={effective['evidence_enabled']}"
        )
    if effective["top_k"] != group.top_k:
        raise EvaluationRunError(
            f"top_k 未按实验配置生效: expected={group.top_k}, actual={effective['top_k']}"
        )


def _success_result(
    *,
    case: EvalCase,
    group_name: str,
    kb_aliases: list[str],
    data: dict[str, Any],
    contexts: list[dict[str, Any]],
    effective_config: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    metadata = data.get("metadata") or {}
    query_processing = metadata.get("query_processing") or {}
    rerank = metadata.get("rerank") or {}
    evidence = metadata.get("evidence") or {}
    graph_injection = metadata.get("graph_injection") or {}
    context_expansion = metadata.get("context") or {}
    calls = int(bool(query_processing.get("enabled")))
    calls += int(bool(rerank.get("enabled")))
    calls += int(bool(evidence.get("executed")))
    return {
        "case_id": case.id,
        "group": group_name,
        "kb_aliases": kb_aliases,
        "question": case.question,
        "ground_truth": case.ground_truth,
        "case_type": case.case_type,
        "tags": case.tags,
        "expected_documents": case.expected_documents,
        "expected_headings": case.expected_headings,
        "answer": None,
        "contexts": contexts,
        "latency_ms": round(latency_ms, 3),
        "stage_latency_ms": {
            "query_processing": query_processing.get("latency_ms", 0),
            "retrieval": metadata.get("latency_ms"),
            "rerank": rerank.get("latency_ms", 0),
            "evidence": evidence.get("latency_ms", 0),
            "graph_injection": graph_injection.get("latency_ms", 0),
            "context_expansion": context_expansion.get("latency_ms", 0),
        },
        "query_processing": {
            "enabled": bool(query_processing.get("enabled")),
            "strategy": query_processing.get("strategy"),
            "raw_query": query_processing.get("raw_query"),
            "effective_query": query_processing.get("effective_query"),
            "search_query": query_processing.get("search_query"),
            "synonym_expansions": query_processing.get("synonym_expansions") or [],
            "degraded": bool(query_processing.get("degraded")),
            "degraded_reason": query_processing.get("degraded_reason"),
        }
        if query_processing
        else None,
        "diagnostics": {
            "graph_injection_executed": bool(graph_injection.get("executed")),
            "graph_injected_count": int(graph_injection.get("injected_count") or 0),
            "supplemental_chunk_count": int(
                context_expansion.get("supplemental_chunk_count") or 0
            ),
            "graph_injection_latency_ms": int(graph_injection.get("latency_ms") or 0),
            "context_expansion_latency_ms": int(context_expansion.get("latency_ms") or 0),
        },
        "application_model_calls": calls,
        "retry_count": 0,
        "effective_config": effective_config,
        "error": None,
    }


def _build_manifest(
    *,
    experiment: ExperimentConfig,
    dataset: EvalDataset,
    started_at: datetime,
    case_count: int,
    status: str,
) -> dict[str, Any]:
    git_commit, dirty = _git_state()
    return {
        "run_id": f"{started_at.strftime('%Y%m%d_%H%M%S')}_{experiment.experiment_id}",
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "dataset_version": dataset.version,
        "corpus_versions": {
            alias: item.corpus_version for alias, item in dataset.knowledge_bases.items()
        },
        "knowledge_base_ids": {
            alias: item.kb_id for alias, item in dataset.knowledge_bases.items()
        },
        "suite": experiment.suite,
        "case_count": case_count,
        "primary_metric": experiment.primary_metric,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "git_commit": git_commit,
        "dirty": dirty,
        "concurrency": experiment.concurrency,
        "ragas_concurrency": experiment.ragas_concurrency,
        "warmup_cases": experiment.warmup_cases,
        "timeout_seconds": experiment.timeout_seconds,
        "baseline": experiment.baseline.model_dump(mode="json"),
        "candidate": experiment.candidate.model_dump(mode="json"),
        "frozen_baseline": (
            {
                "run_dir": experiment.baseline_run_dir,
                "group": experiment.baseline_group,
            }
            if experiment.baseline_run_dir
            else None
        ),
        "telemetry": {
            "ragas_do_not_track": True,
            "offline_langfuse_enabled": False,
        },
        "index_policy": {
            "baseline": experiment.baseline.index_version,
            "candidate": experiment.candidate.index_version,
        },
        "status": status,
    }


def _create_run_dir(
    *,
    output_root: Path,
    experiment_id: str,
    output_suffix: str | None,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{output_suffix}" if output_suffix else ""
    run_dir = output_root / f"{timestamp}_{experiment_id}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _normalize_document_name(value: str) -> str:
    name = Path(value).name.strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    return name.strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
