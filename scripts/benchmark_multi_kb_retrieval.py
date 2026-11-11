#!/usr/bin/env python3
"""Benchmark fixed retrieval requests across one- and multi-KB scopes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import load_dataset
from scripts.run_retrieval_eval import (
    evaluation_api_key,
    load_request_config,
    resolve_case_kb_ids,
    retrieve_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1库/多库固定检索性能基线")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--request-config", type=Path, required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat 必须大于0，warmup 不能小于0")
    dataset = load_dataset(args.dataset)
    request_config = load_request_config(args.request_config)
    cases = dataset.get("cases") or []
    default_kb_id = str(dataset.get("kb_id") or "").strip()
    profile = str(request_config.get("profile") or dataset.get("default_profile") or "custom")
    if not cases or not default_kb_id:
        raise ValueError("数据集必须包含 kb_id 和 cases")

    samples: list[dict[str, Any]] = []
    with evaluation_api_key(args) as api_key:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            for iteration in range(1, args.warmup + args.repeat + 1):
                is_warmup = iteration <= args.warmup
                for case in cases:
                    case_id = str(case.get("id") or case.get("question") or "case")
                    question = str(case.get("question") or "").strip()
                    kb_ids = resolve_case_kb_ids(case, default_kb_id)
                    started_at = perf_counter()
                    try:
                        data = retrieve_case(
                            client,
                            base_url=args.base_url,
                            api_key=api_key,
                            kb_ids=kb_ids,
                            question=question,
                            profile=profile,
                            request_config=request_config,
                        )
                        latency_ms = (perf_counter() - started_at) * 1000
                        chunks = data.get("retrieved_chunks") or []
                        _assert_frozen_response(chunks, request_config)
                        source_counts: dict[str, int] = {}
                        for chunk in chunks:
                            source_kb_id = str(chunk.get("kb_id") or "")
                            source_counts[source_kb_id] = source_counts.get(source_kb_id, 0) + 1
                        samples.append(
                            {
                                "iteration": iteration,
                                "warmup": is_warmup,
                                "case_id": case_id,
                                "kb_count": len(kb_ids),
                                "kb_ids": kb_ids,
                                "latency_ms": round(latency_ms, 3),
                                "returned_count": len(chunks),
                                "source_counts": source_counts,
                                "metadata_latency_ms": (data.get("metadata") or {}).get("latency_ms"),
                                "error": None,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - persist each failed sample
                        latency_ms = (perf_counter() - started_at) * 1000
                        samples.append(
                            {
                                "iteration": iteration,
                                "warmup": is_warmup,
                                "case_id": case_id,
                                "kb_count": len(kb_ids),
                                "kb_ids": kb_ids,
                                "latency_ms": round(latency_ms, 3),
                                "returned_count": 0,
                                "source_counts": {},
                                "metadata_latency_ms": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                label = "warmup" if is_warmup else "measured"
                print(f"{label} iteration {iteration}/{args.warmup + args.repeat} complete")

    measured = [sample for sample in samples if not sample["warmup"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "request_config_path": str(args.request_config),
        "request_config": request_config,
        "parameters": {
            "repeat_per_case": args.repeat,
            "warmup_per_case": args.warmup,
            "case_count": len(cases),
            "request_concurrency": 1,
        },
        "summary_by_kb_count": _summarize(measured),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary_by_kb_count"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")
    return 2 if any(sample["error"] for sample in measured) else 0


def _assert_frozen_response(
    chunks: list[dict[str, Any]],
    request_config: dict[str, Any],
) -> None:
    expected_top_k = int(request_config.get("top_k") or 0)
    if len(chunks) != expected_top_k:
        raise ValueError(f"返回数量不符合冻结配置: expected={expected_top_k}, actual={len(chunks)}")
    expected_version = str(request_config.get("index_version") or "")
    actual_versions = {str(chunk.get("index_version") or "") for chunk in chunks}
    if actual_versions != {expected_version}:
        raise ValueError(
            "索引版本不符合冻结配置: "
            f"expected={expected_version}, actual={sorted(actual_versions)}"
        )


def _summarize(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["kb_count"]), []).append(sample)
    summary: dict[str, dict[str, Any]] = {}
    for kb_count, group in sorted(grouped.items()):
        successful = [sample for sample in group if sample["error"] is None]
        latencies = [float(sample["latency_ms"]) for sample in successful]
        balanced_source_cases = sum(
            len(set(sample.get("source_counts", {}).values())) == 1
            and len(sample.get("source_counts", {})) > 1
            for sample in successful
        )
        summary[str(kb_count)] = {
            "sample_count": len(group),
            "successful_samples": len(successful),
            "failed_samples": len(group) - len(successful),
            "latency_mean_ms": mean(latencies) if latencies else None,
            "latency_p50_ms": median(latencies) if latencies else None,
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_max_ms": max(latencies) if latencies else None,
            "equal_source_allocation_samples": balanced_source_cases,
        }
    return summary


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    raise SystemExit(main())
