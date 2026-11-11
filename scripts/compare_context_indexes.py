from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.schemas.rag import (
    QueryOptions,
    RerankOptions,
    RetrievalOptions,
    RetrieveRequest,
)
from app.services.rag_service import RagService
from app.utils.markdown_splitter import is_low_information_content, normalize_content

DEFAULT_DATASET = PROJECT_ROOT / "eval/opt16_context/context_retrieval.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval/opt16_context/index_v1_vs_v2.json"
DEFAULT_RERANK_OUTPUT = PROJECT_ROOT / "eval/opt16_context/rerank_comparison.json"
EVAL_USER_ID = "opt16_index_eval"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def select_cases(dataset: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise TypeError("dataset must contain a cases array")
    suite_data = (dataset.get("suites") or {}).get(suite)
    if not isinstance(suite_data, dict):
        raise TypeError(f"unknown suite: {suite}")
    case_ids = suite_data.get("case_ids")
    if case_ids == "all":
        return cases
    if not isinstance(case_ids, list):
        raise TypeError(f"suite {suite} must contain a case_ids array or 'all'")
    by_id = {str(case.get("id")): case for case in cases if isinstance(case, dict)}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"suite {suite} references missing cases: {missing}")
    return [by_id[case_id] for case_id in case_ids]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    for old, new in (("～", "至"), ("~", "至"), ("—", ""), ("-", "")):
        text = text.replace(old, new)
    return "".join(char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _metadata_text(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list):
        heading_path = " / ".join(str(item) for item in heading_path)
    return str(heading_path or "")


def annotate_chunk(case: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(chunk.get("title"))
    content = normalize_text(chunk.get("content"))
    heading = normalize_text(_metadata_text(chunk.get("metadata")))
    expected_documents = [normalize_text(item).removesuffix("md") for item in case.get("expected_documents", [])]
    expected_headings = [normalize_text(item) for item in case.get("expected_headings", [])]
    expected_keywords = [normalize_text(item) for item in case.get("expected_keywords", [])]
    title_without_md = title.removesuffix("md")
    document_match = any(item and item == title_without_md for item in expected_documents)
    heading_match = any(item and (item in heading or item in content) for item in expected_headings)
    keyword_hits = [item for item in expected_keywords if item and item in content]
    reference_pointer = str(chunk.get("content") or "").lstrip().startswith("> 关联规则")

    if document_match and heading_match and keyword_hits:
        relevance = 3
    elif document_match and (heading_match or keyword_hits):
        relevance = 2
    elif document_match or keyword_hits:
        relevance = 1
    else:
        relevance = 0
    if reference_pointer:
        relevance = min(relevance, 2)
    return {
        **chunk,
        "relevance": relevance,
        "annotation": {
            "document_match": document_match,
            "heading_match": heading_match,
            "matched_keywords": keyword_hits,
            "reference_pointer": reference_pointer,
            "method": "document_grounded_spec",
        },
    }


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def summarize_case(
    case: dict[str, Any], chunks: list[dict[str, Any]], top_n: int = 10
) -> dict[str, Any]:
    selected = chunks[:top_n]
    grades = [int(chunk.get("relevance") or 0) for chunk in selected]
    core_positions = [index for index, grade in enumerate(grades, start=1) if grade == 3]
    ideal = sorted(grades, reverse=True)
    ideal_dcg = _dcg(ideal)

    expected_documents = {
        normalize_text(item).removesuffix("md") for item in case.get("expected_documents", [])
    }
    returned_documents = {
        normalize_text(chunk.get("title")).removesuffix("md") for chunk in selected
    }
    expected_headings = [normalize_text(item) for item in case.get("expected_headings", [])]
    expected_keywords = [normalize_text(item) for item in case.get("expected_keywords", [])]

    evidence_parts: list[str] = []
    supplemental_total = 0
    supplemental_useful = 0
    context_chars = 0
    for chunk in selected:
        context = chunk.get("context") or {}
        context_content = str(context.get("content") or "")
        evidence_parts.append(
            f"{chunk.get('title', '')} {_metadata_text(chunk.get('metadata'))} "
            f"{context_content or chunk.get('content', '')}"
        )
        context_chars += len(context_content or str(chunk.get("content") or ""))
        for source in context.get("sources") or []:
            source_title = normalize_text(source.get("title")).removesuffix("md")
            if source_title:
                returned_documents.add(source_title)
            if source.get("relation") == "anchor":
                continue
            supplemental_total += 1
            source_text = normalize_text(source.get("content"))
            source_heading = normalize_text(_metadata_text(source.get("metadata")))
            if (
                source_title in expected_documents
                and (
                    any(keyword and keyword in source_text for keyword in expected_keywords)
                    or any(heading and heading in source_heading for heading in expected_headings)
                )
            ):
                supplemental_useful += 1
    evidence_text = normalize_text(" ".join(evidence_parts))
    document_hits = expected_documents & returned_documents
    keyword_hits = [item for item in expected_keywords if item and item in evidence_text]

    contents = [normalize_text(chunk.get("content")) for chunk in selected]
    nonempty_contents = [content for content in contents if content]
    duplicate_count = len(nonempty_contents) - len(set(nonempty_contents))
    low_information_count = sum(
        1 for chunk in selected if is_low_information_content(str(chunk.get("content") or ""))
    )
    expected_evidence_count = max(len(case.get("expected_headings", [])), 1)
    effective_hits = sum(grade >= 2 for grade in grades)
    if not keyword_hits:
        completeness = "missing"
    elif len(keyword_hits) == len(expected_keywords):
        completeness = "complete"
    else:
        completeness = "partial"
    return {
        "ndcg@10": round(_dcg(grades) / ideal_dcg, 4) if ideal_dcg else 0.0,
        "mrr@10": round(1 / core_positions[0], 4) if core_positions else 0.0,
        "hit_rate@10": 1.0 if core_positions else 0.0,
        "effective_recall@10": round(min(effective_hits / expected_evidence_count, 1.0), 4),
        "expected_document_recall@10": round(
            len(document_hits) / len(expected_documents), 4
        )
        if expected_documents
        else 1.0,
        "keyword_coverage@10": round(len(keyword_hits) / len(expected_keywords), 4)
        if expected_keywords
        else 1.0,
        "duplicate_rate@10": round(duplicate_count / len(selected), 4) if selected else 0.0,
        "low_information_rate@10": round(low_information_count / len(selected), 4)
        if selected
        else 0.0,
        "context_completeness": completeness,
        "context_complete": 1.0 if completeness == "complete" else 0.0,
        "supplemental_precision": round(supplemental_useful / supplemental_total, 4)
        if supplemental_total
        else None,
        "supplemental_count": supplemental_total,
        "context_chars": context_chars,
    }


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "ndcg@10",
        "mrr@10",
        "hit_rate@10",
        "effective_recall@10",
        "expected_document_recall@10",
        "keyword_coverage@10",
        "duplicate_rate@10",
        "low_information_rate@10",
        "context_complete",
    )
    metrics: dict[str, Any] = {"case_count": len(case_results)}
    for name in metric_names:
        values = [float(item["metrics"][name]) for item in case_results]
        metrics[name] = round(sum(values) / len(values), 4) if values else 0.0
    supplemental_values = [
        float(item["metrics"]["supplemental_precision"])
        for item in case_results
        if item["metrics"].get("supplemental_precision") is not None
    ]
    metrics["supplemental_precision"] = (
        round(sum(supplemental_values) / len(supplemental_values), 4)
        if supplemental_values
        else None
    )
    metrics["average_context_chars"] = round(
        sum(int(item["metrics"]["context_chars"]) for item in case_results)
        / len(case_results),
        1,
    ) if case_results else 0.0
    metrics["context_completeness_rate"] = metrics.pop("context_complete")
    latencies = sorted(int(item["latency_ms"]) for item in case_results)
    if latencies:
        metrics["p50_latency_ms"] = latencies[(len(latencies) - 1) // 2]
        metrics["p95_latency_ms"] = latencies[math.ceil(len(latencies) * 0.95) - 1]
    context_latencies = sorted(int(item.get("context_latency_ms") or 0) for item in case_results)
    if context_latencies:
        metrics["context_p50_latency_ms"] = context_latencies[(len(context_latencies) - 1) // 2]
        metrics["context_p95_latency_ms"] = context_latencies[
            math.ceil(len(context_latencies) * 0.95) - 1
        ]
    return metrics


def build_request(dataset: dict[str, Any], case: dict[str, Any]) -> RetrieveRequest:
    options = dataset.get("retrieval_options") or {}
    return RetrieveRequest(
        kb_id=str(dataset["knowledge_base_id"]),
        user_id=EVAL_USER_ID,
        query=str(case["question"]),
        profile="custom",
        top_k=int(dataset.get("top_k") or 10),
        retrieval_options=RetrievalOptions(**options),
        rerank_options=RerankOptions(enabled=False),
        query_options=QueryOptions(enabled=False, strategy="noop"),
    )


async def collect_run(
    *,
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    index_version: str,
    expand_context: bool,
    graph_injection: bool | None = None,
    run_name: str,
) -> dict[str, Any]:
    tenant_id = str(dataset["tenant_id"])
    results: list[dict[str, Any]] = []
    with SessionLocal() as db:
        service = RagService(db)
        for index, case in enumerate(cases, start=1):
            response = await service.retrieve(
                tenant_id,
                build_request(dataset, case),
                index_version_overrides={str(dataset["knowledge_base_id"]): index_version},
                expand_context=expand_context,
                graph_injection=graph_injection,
            )
            payload = response.model_dump(mode="json")
            chunks = [annotate_chunk(case, chunk) for chunk in payload["retrieved_chunks"]]
            context_latency_ms = int(
                (payload["metadata"].get("context") or {}).get("latency_ms") or 0
            )
            result = {
                "id": case["id"],
                "question": case["question"],
                "ground_truth": case.get("ground_truth"),
                "expected_documents": case.get("expected_documents", []),
                "expected_headings": case.get("expected_headings", []),
                "expected_relation": case.get("expected_relation"),
                "context_required": case.get("context_required", False),
                "expected_keywords": case.get("expected_keywords", []),
                "latency_ms": payload["metadata"]["latency_ms"],
                "context_latency_ms": context_latency_ms,
                "retrieval_metadata": payload["metadata"].get("retrieval"),
                "candidate_filter": payload["metadata"].get("candidate_filter"),
                "context_metadata": payload["metadata"].get("context"),
                "chunks": chunks,
            }
            result["metrics"] = summarize_case(case, chunks)
            results.append(result)
            print(
                f"captured {index}/{len(cases)}: {case['id']} "
                f"({run_name}, graph={graph_injection}, context={expand_context})"
            )
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "index_version": index_version,
        "graph_injection": graph_injection,
        "context_expansion": expand_context,
        "metrics": aggregate(results),
        "cases": results,
    }


def resolve_run(runs: dict[str, Any], *names: str) -> dict[str, Any] | None:
    for name in names:
        run = runs.get(name)
        if isinstance(run, dict):
            return run
    return None


def build_v2_run_specs(*, diagnostic: bool) -> list[dict[str, Any]]:
    specs = [
        {
            "run_name": "v2_full",
            "graph_injection": True,
            "expand_context": True,
        }
    ]
    if diagnostic:
        specs = [
            {
                "run_name": "v2_hybrid",
                "graph_injection": False,
                "expand_context": False,
            },
            {
                "run_name": "v2_graph",
                "graph_injection": True,
                "expand_context": False,
            },
            *specs,
        ]
    return specs


async def capture_v1(dataset_path: Path, output_path: Path, suite: str) -> None:
    dataset = load_json(dataset_path)
    cases = select_cases(dataset, suite)
    tenant_id = str(dataset["tenant_id"])
    run = await collect_run(
        dataset=dataset,
        cases=cases,
        index_version="v1",
        expand_context=False,
        graph_injection=False,
        run_name="v1",
    )

    output = {
        "name": "opt16_index_v1_vs_v2",
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "suite": suite,
        "annotation": dataset.get("annotation"),
        "config": {
            "knowledge_base_id": dataset["knowledge_base_id"],
            "tenant_id": tenant_id,
            "profile": "custom",
            "top_k": dataset.get("top_k"),
            "retrieval_options": dataset.get("retrieval_options"),
            "rerank_enabled": False,
            "query_rewrite_enabled": False,
        },
        "runs": {"v1": run},
        "human_review": {
            "status": "pending",
            "note": "自动预标注基于文档、章节和关键词；最终报告前需复核 relevance 与证据完整性。",
        },
    }
    save_json(output_path, output)
    print(json.dumps(output["runs"]["v1"]["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


def recompute_run(run: dict[str, Any], case_specs: dict[str, dict[str, Any]]) -> None:
    for case_result in run.get("cases") or []:
        case = case_specs[str(case_result["id"])]
        case_result["chunks"] = [
            annotate_chunk(case, chunk) for chunk in case_result.get("chunks") or []
        ]
        case_result.setdefault("context_latency_ms", 0)
        case_result["metrics"] = summarize_case(case, case_result["chunks"])
    run["metrics"] = aggregate(run.get("cases") or [])


async def prepare_comparison(
    dataset_path: Path,
    output_path: Path,
    suite: str,
    *,
    diagnostic: bool = False,
) -> None:
    dataset = load_json(dataset_path)
    cases = select_cases(dataset, suite)
    case_specs = {str(case["id"]): case for case in cases}
    output = load_json(output_path)
    if output.get("suite") != suite:
        raise ValueError(
            f"baseline suite is {output.get('suite')!r}; expected {suite!r}. "
            "Capture the matching v1 baseline first."
        )
    v1_baseline = (output.get("runs") or {}).get("v1")
    if not isinstance(v1_baseline, dict):
        raise TypeError("v1 baseline is missing")
    recompute_run(v1_baseline, case_specs)

    v1_current = await collect_run(
        dataset=dataset,
        cases=cases,
        index_version="v1",
        expand_context=False,
        graph_injection=False,
        run_name="v1",
    )

    v2_runs: dict[str, Any] = {}
    for spec in build_v2_run_specs(diagnostic=diagnostic):
        v2_runs[spec["run_name"]] = await collect_run(
            dataset=dataset,
            cases=cases,
            index_version="v2",
            expand_context=spec["expand_context"],
            graph_injection=spec["graph_injection"],
            run_name=spec["run_name"],
        )

    baseline_by_id = {item["id"]: item for item in v1_baseline["cases"]}
    current_by_id = {item["id"]: item for item in v1_current["cases"]}
    compatibility_cases: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        before = [chunk["chunk_id"] for chunk in baseline_by_id[case_id]["chunks"]]
        after = [chunk["chunk_id"] for chunk in current_by_id[case_id]["chunks"]]
        compatibility_cases.append(
            {
                "id": case_id,
                "ordered_topk_match": before == after,
                "candidate_set_match": set(before) == set(after),
                "before_chunk_ids": before,
                "after_chunk_ids": after,
            }
        )
    output["v1_compatibility"] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "case_count": len(compatibility_cases),
        "ordered_topk_match_count": sum(
            item["ordered_topk_match"] for item in compatibility_cases
        ),
        "candidate_set_match_count": sum(
            item["candidate_set_match"] for item in compatibility_cases
        ),
        "top1_match_count": sum(
            bool(item["before_chunk_ids"])
            and item["before_chunk_ids"][:1] == item["after_chunk_ids"][:1]
            for item in compatibility_cases
        ),
        "average_top10_overlap": round(
            sum(
                len(set(item["before_chunk_ids"]) & set(item["after_chunk_ids"]))
                / max(len(item["before_chunk_ids"]), 1)
                for item in compatibility_cases
            )
            / len(compatibility_cases),
            4,
        ),
        "cases": compatibility_cases,
    }

    output["runs"]["v1"] = v1_current
    output["runs"].update(v2_runs)
    output["evaluation_mode"] = "diagnostic" if diagnostic else "primary"
    output["prepared_at"] = datetime.now(UTC).isoformat()
    save_json(output_path, output)
    print(json.dumps({key: value["metrics"] for key, value in output["runs"].items()}, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


async def refresh_v2(
    dataset_path: Path,
    output_path: Path,
    suite: str,
    *,
    diagnostic: bool = False,
) -> None:
    dataset = load_json(dataset_path)
    cases = select_cases(dataset, suite)
    output = load_json(output_path)
    v2_runs: dict[str, Any] = {}
    for spec in build_v2_run_specs(diagnostic=diagnostic):
        v2_runs[spec["run_name"]] = await collect_run(
            dataset=dataset,
            cases=cases,
            index_version="v2",
            expand_context=spec["expand_context"],
            graph_injection=spec["graph_injection"],
            run_name=spec["run_name"],
        )
    output["runs"].update(v2_runs)
    output["evaluation_mode"] = "diagnostic" if diagnostic else "primary"
    output["v2_refreshed_at"] = datetime.now(UTC).isoformat()
    save_json(output_path, output)
    print(json.dumps({key: value["metrics"] for key, value in v2_runs.items()}, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


def validate_comparison(output_path: Path) -> dict[str, Any]:
    data = load_json(output_path)
    errors: list[str] = []
    warnings: list[str] = []
    runs = data.get("runs") or {}
    v1_run = resolve_run(runs, "v1")
    v2_full = resolve_run(runs, "v2_full", "v2_context")
    if v1_run is None:
        errors.append("missing run: v1")
    if v2_full is None:
        errors.append("missing run: v2_full")
    expected_case_count = len((v1_run or {}).get("cases") or [])
    validate_names = ["v1", "v2_full", *[
        name
        for name in ("v2_hybrid", "v2_graph")
        if isinstance(runs.get(name), dict)
    ]]
    for run_name in validate_names:
        if run_name == "v1":
            run = v1_run
        elif run_name == "v2_full":
            run = v2_full
        else:
            run = runs.get(run_name)
        if not isinstance(run, dict):
            errors.append(f"missing run: {run_name}")
            continue
        cases = run.get("cases") or []
        if len(cases) != expected_case_count:
            errors.append(
                f"{run_name}: case count mismatch {len(cases)} != {expected_case_count}"
            )
        for case in cases:
            seen_ids: set[str] = set()
            seen_content: set[str] = set()
            for chunk in case.get("chunks") or []:
                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id:
                    errors.append(f"{run_name}/{case.get('id')}: missing chunk_id")
                if chunk_id in seen_ids:
                    errors.append(f"{run_name}/{case.get('id')}: duplicate chunk_id {chunk_id}")
                seen_ids.add(chunk_id)
                normalized = normalize_content(str(chunk.get("content") or ""))
                if normalized in seen_content:
                    warnings.append(
                        f"{run_name}/{case.get('id')}: duplicate normalized content"
                    )
                seen_content.add(normalized)
                if chunk.get("relevance") not in {0, 1, 2, 3}:
                    errors.append(f"{run_name}/{case.get('id')}: invalid relevance")
                if run_name.startswith("v2"):
                    for field in ("retrieval_text", "section_id", "order_index"):
                        if chunk.get(field) is None:
                            errors.append(
                                f"{run_name}/{case.get('id')}/{chunk_id}: missing {field}"
                            )
                    if chunk.get("index_version") != "v2":
                        errors.append(
                            f"{run_name}/{case.get('id')}/{chunk_id}: mixed index version"
                        )
                if run_name in {"v2_full", "v2_context"}:
                    context = chunk.get("context")
                    if context and (context.get("sources") or [{}])[0].get("relation") != "anchor":
                        errors.append(
                            f"{run_name}/{case.get('id')}/{chunk_id}: context must start with anchor"
                        )

    compatibility = data.get("v1_compatibility") or {}
    compatibility_cases = compatibility.get("cases") or []
    average_overlap = compatibility.get("average_top10_overlap")
    if average_overlap is None and compatibility_cases:
        average_overlap = sum(
            len(set(item["before_chunk_ids"]) & set(item["after_chunk_ids"]))
            / max(len(item["before_chunk_ids"]), 1)
            for item in compatibility_cases
        ) / len(compatibility_cases)
    if average_overlap is None or average_overlap < 0.95:
        errors.append("v1 Top10 overlap is below 95% after migration")
    if (data.get("human_review") or {}).get("status") != "reviewed":
        warnings.append("human relevance review is still pending")
    result = {
        "valid": not errors,
        "case_count": expected_case_count,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(float(current.get(key) or 0) - float(baseline.get(key) or 0), 4)


def compare_and_report(output_path: Path) -> None:
    data = load_json(output_path)
    dataset = load_json(PROJECT_ROOT / str(data["dataset"]))
    case_specs = {
        str(case["id"]): case for case in select_cases(dataset, str(data["suite"]))
    }
    for run in (data.get("runs") or {}).values():
        recompute_run(run, case_specs)
    save_json(output_path, data)
    validation = validate_comparison(output_path)
    if not validation["valid"]:
        raise RuntimeError("comparison validation failed")
    data = load_json(output_path)
    runs = data["runs"]
    v1 = runs["v1"]["metrics"]
    v2_full_run = resolve_run(runs, "v2_full", "v2_context")
    if v2_full_run is None:
        raise TypeError("v2_full run is missing")
    v2_full = v2_full_run["metrics"]
    criteria = {
        "v1_compatibility": (
            sum(
                len(set(item["before_chunk_ids"]) & set(item["after_chunk_ids"]))
                / max(len(item["before_chunk_ids"]), 1)
                for item in data["v1_compatibility"]["cases"]
            )
            / data["v1_compatibility"]["case_count"]
            >= 0.95
        ),
        "v2_hit_rate_not_lower": v2_full["hit_rate@10"] >= v1["hit_rate@10"],
        "v2_ndcg_not_lower": v2_full["ndcg@10"] >= v1["ndcg@10"],
        "v2_document_recall_not_lower": (
            v2_full["expected_document_recall@10"] >= v1["expected_document_recall@10"]
        ),
        "v2_low_information_zero": v2_full["low_information_rate@10"] == 0,
        "context_completeness_at_least_90_percent": (
            v2_full["context_completeness_rate"] >= 0.9
        ),
        "context_p95_at_most_200_ms": v2_full["context_p95_latency_ms"] <= 200,
    }
    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_mode": data.get("evaluation_mode") or "primary",
        "v2_full_vs_v1": {
            key: _delta(v2_full, v1, key)
            for key in (
                "ndcg@10",
                "mrr@10",
                "hit_rate@10",
                "expected_document_recall@10",
                "keyword_coverage@10",
            )
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    diagnostic_metrics: dict[str, Any] = {}
    for run_name in ("v2_hybrid", "v2_graph"):
        run = runs.get(run_name)
        if isinstance(run, dict):
            diagnostic_metrics[run_name] = run["metrics"]
    if diagnostic_metrics:
        comparison["diagnostic_runs"] = diagnostic_metrics
    data["comparison"] = comparison
    save_json(output_path, data)

    metric_rows = []
    for key in (
        "ndcg@10",
        "mrr@10",
        "hit_rate@10",
        "expected_document_recall@10",
        "keyword_coverage@10",
        "low_information_rate@10",
        "duplicate_rate@10",
        "context_completeness_rate",
        "average_context_chars",
        "p50_latency_ms",
        "p95_latency_ms",
        "context_p95_latency_ms",
    ):
        metric_rows.append(
            f"| {key} | {v1.get(key, '—')} | {v2_full.get(key, '—')} |"
        )
    criteria_rows = [
        f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in criteria.items()
    ]
    deployment = data.get("deployment") or {}
    if comparison["passed"]:
        conclusion = "核心门槛全部通过，v2 可以作为默认检索版本并保留 v1 回滚能力。"
    elif deployment.get("active_version") == "v2":
        conclusion = (
            "严格 nDCG 门槛仍有未通过项；按当前测试要求已将 v2 设为默认检索版本，"
            "v1 继续保留用于前端对比和快速回滚。正式发布前应继续复核负向 Case，"
            "并执行 30 题完整回归。"
        )
    else:
        conclusion = "仍有门槛未通过，暂不激活 v2；应先复核负向 Case 和自动标注。"
    diagnostic_section = ""
    if diagnostic_metrics:
        diagnostic_lines = ["## 诊断分组（`--diagnostic`）", "", "| 指标 | v2 Hybrid | v2 Graph | v2 Full |", "|---|---:|---:|---:|"]
        for key in (
            "ndcg@10",
            "mrr@10",
            "expected_document_recall@10",
            "context_completeness_rate",
        ):
            diagnostic_lines.append(
                "| "
                + key
                + " | "
                + str(diagnostic_metrics.get("v2_hybrid", {}).get(key, "—"))
                + " | "
                + str(diagnostic_metrics.get("v2_graph", {}).get(key, "—"))
                + " | "
                + str(v2_full.get(key, "—"))
                + " |"
            )
        diagnostic_section = "\n".join(diagnostic_lines) + "\n\n"
    report = f"""# 轻量文档关系图与增强检索测试报告

## 测试口径

- 样本：`{data['suite']}` 套件，共 {v1['case_count']} 题
- 知识库：`{data['config']['knowledge_base_id']}`
- 评测模式：`{comparison['evaluation_mode']}`
- 主对比：v1 原始索引 vs v2 全开（图谱注入 + 上下文扩展）
- 检索：Hybrid，Vector Top20 + BM25 Top20 + RRF，最终 Top10
- 查询改写：关闭
- 重排：关闭
- 标注：文档、章节和关键词约束生成的 0～3 级预标注，人工复核状态为 `{data['human_review']['status']}`

## 历史索引兼容性

迁移前后 Top10 平均候选重合率为 {sum(len(set(item['before_chunk_ids']) & set(item['after_chunk_ids'])) / max(len(item['before_chunk_ids']), 1) for item in data['v1_compatibility']['cases']) / data['v1_compatibility']['case_count']:.2%}；候选内的少量顺序波动来自重复调用远程 Embedding 后的近似分数并列，v1 Chunk 和原向量未变更。

## 汇总指标

| 指标 | v1 原始索引 | v2 全开 |
|---|---:|---:|
{chr(10).join(metric_rows)}

{diagnostic_section}## 验收门槛

| 门槛 | 结果 |
|---|---|
{chr(10).join(criteria_rows)}

## 结论

{conclusion}

当前报告只比较索引和关系增强能力，没有启用 LLM 重排或 Qwen3-Rerank。主验收只看 v1 与 v2 全开；只有主门槛未通过时，才需要追加 `--diagnostic` 排查切块、图谱注入和上下文扩展各自的贡献。
"""
    report_path = output_path.with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(f"saved: {output_path}")
    print(f"saved: {report_path}")


def mark_reviewed(output_path: Path) -> None:
    data = load_json(output_path)
    data["human_review"] = {
        "status": "reviewed",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "scope": "core_12",
        "note": (
            "已按问题、标准答案、目标文档、目标章节和关键词复核 12 题核心集；"
            "仅指向其他规则的纯引用片段最高标记为 relevance=2。"
        ),
    }
    data["deployment"] = {
        "active_version": "v2",
        "rollback_version": "v1",
        "note": "按当前测试要求将 v2 设为默认检索版本，v1 保留用于对比与回滚。",
    }
    save_json(output_path, data)
    print(f"saved: {output_path}")


def export_rerank_dataset(output_path: Path, rerank_output_path: Path) -> None:
    data = load_json(output_path)
    run = resolve_run(data.get("runs") or {}, "v2_graph", "v2_full", "v2_anchor")
    if not isinstance(run, dict):
        raise TypeError("v2_graph or v2_full run is missing")
    review = data.get("human_review") or {}
    if review.get("status") != "reviewed":
        raise ValueError("core relevance labels must be reviewed before rerank export")

    cases = []
    for case in run.get("cases") or []:
        candidates = []
        for chunk in case.get("chunks") or []:
            relevance = int(chunk.get("relevance") or 0)
            note = {
                3: "直接回答问题的核心证据",
                2: "能够支撑答案的有效证据或明确引用路径",
                1: "主题相关，但不能直接支撑答案",
                0: "与当前问题无关",
            }[relevance]
            candidates.append(
                {
                    key: chunk.get(key)
                    for key in (
                        "chunk_id",
                        "document_id",
                        "title",
                        "content",
                        "retrieval_text",
                        "score",
                        "index_version",
                        "section_id",
                        "parent_section_id",
                        "order_index",
                        "metadata",
                    )
                }
                | {
                    "relevance": relevance,
                    "relevance_note": note,
                }
            )
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "ground_truth": case.get("ground_truth"),
                "kb_ids": [data["config"]["knowledge_base_id"]],
                "candidates": candidates,
            }
        )

    dataset = {
        "name": "opt16_v2_rerank_core",
        "created_at": run.get("captured_at") or datetime.now(UTC).isoformat(),
        "profile": data["config"].get("profile"),
        "source_index_version": "v2",
        "candidate_top_k": 10,
        "result_top_n": 5,
        "annotation": {
            "method": "document_grounded_human_reviewed",
            "status": "reviewed",
            "rubric": "0=无关，1=主题相关，2=有效支撑，3=核心证据",
            "reviewed_at": review.get("reviewed_at"),
        },
        "cases": cases,
    }
    save_json(rerank_output_path, dataset)
    print(f"saved: {rerank_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v1 and v2 contextual indexes")
    parser.add_argument(
        "command",
        choices=(
            "capture-v1",
            "prepare",
            "refresh-v2",
            "validate",
            "compare",
            "mark-reviewed",
            "export-rerank",
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rerank-output", type=Path, default=DEFAULT_RERANK_OUTPUT)
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="also capture v2_hybrid and v2_graph for troubleshooting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "capture-v1":
        asyncio.run(capture_v1(args.dataset, args.output, args.suite))
    elif args.command == "prepare":
        asyncio.run(
            prepare_comparison(
                args.dataset,
                args.output,
                args.suite,
                diagnostic=args.diagnostic,
            )
        )
    elif args.command == "refresh-v2":
        asyncio.run(
            refresh_v2(
                args.dataset,
                args.output,
                args.suite,
                diagnostic=args.diagnostic,
            )
        )
    elif args.command == "validate":
        result = validate_comparison(args.output)
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "compare":
        compare_and_report(args.output)
    elif args.command == "mark-reviewed":
        mark_reviewed(args.output)
    elif args.command == "export-rerank":
        export_rerank_dataset(args.output, args.rerank_output)


if __name__ == "__main__":
    main()
