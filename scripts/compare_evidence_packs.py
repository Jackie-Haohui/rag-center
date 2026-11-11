from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.schemas.rag import EvidenceOptions
from app.services.evidence_orchestration_service import (
    EvidenceOrchestrationService,
    prompt_sha256,
)

CHALLENGE_FIVE_ASPECTS = {
    "live_category_qualification": [
        {
            "aspect": "达人直播带货权限",
            "acceptable_chunk_ids": ["6400b836-5cf1-43fc-89b8-a2f57e5629d2"],
        },
        {
            "aspect": "美妆类目具体资质",
            "acceptable_chunk_ids": ["44159d28-676b-40c1-a84f-34b8e4ac6c18"],
        },
    ],
    "live_campaign_shipping": [
        {
            "aspect": "直播大场订单适用活动发货规则",
            "acceptable_chunk_ids": ["c4c6be7a-ef36-4a1d-99e9-eff07b442873"],
        },
        {
            "aspect": "大促订单48小时揽收",
            "acceptable_chunk_ids": [
                "413ec4ca-4474-4c54-a97e-ee642182aa66",
                "be592533-2534-499a-ad02-ed0888908d52",
            ],
        },
        {
            "aspect": "延迟发货赔付规则",
            "acceptable_chunk_ids": ["be592533-2534-499a-ad02-ed0888908d52"],
        },
    ],
    "search_ad_invoice_reconciliation": [
        {
            "aspect": "广告账单差异申诉期限",
            "acceptable_chunk_ids": ["079529f9-bae4-48d6-8917-8d3582ed77da"],
        },
        {
            "aspect": "平台开具广告费发票",
            "acceptable_chunk_ids": ["cb269192-7e36-4a31-9832-174fee8ad306"],
        },
        {
            "aspect": "商家结算对账入口与期限",
            "acceptable_chunk_ids": ["da6c6060-b3d8-4966-bc72-129292455044"],
        },
    ],
    "member_points_refund": [
        {
            "aspect": "积分退款及过期处理",
            "acceptable_chunk_ids": ["779b77ca-dcb0-4de1-9eaf-d2279a876f34"],
        },
        {
            "aspect": "现金原支付渠道退回",
            "acceptable_chunk_ids": ["2af0bac1-6126-4d91-bc42-774ca90d9d4d"],
        },
    ],
    "aftersales_intercept_refund": [
        {
            "aspect": "拦截成功和失败后的处理",
            "acceptable_chunk_ids": ["73703d96-4fda-4cab-a2da-7be08461020b"],
        },
        {
            "aspect": "拦截失败后的退货质检退款",
            "acceptable_chunk_ids": ["44d9f1c9-99b3-47b0-8a89-927da0bbb9c0"],
        },
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def successful_qwen_topk(result: dict[str, Any], case_id: str, top_k: int = 5) -> list[str]:
    qwen_cases = result.get("runs", {}).get("qwen3", {}).get("cases", [])
    case = next((item for item in qwen_cases if item.get("id") == case_id), None)
    if not case:
        raise ValueError(f"qwen3 result missing case: {case_id}")
    runs = case.get("runs") or case.get("warmups") or []
    run = next((item for item in runs if item.get("success") and item.get("rankings")), None)
    if not run:
        raise ValueError(f"qwen3 result has no successful rankings: {case_id}")
    return [str(item["chunk_id"]) for item in run["rankings"][:top_k]]


def derive_required_aspects(case: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for candidate in case["candidates"]:
        if candidate.get("relevance") != 3:
            continue
        metadata = candidate.get("metadata") or {}
        heading = str(metadata.get("heading_path") or candidate.get("title") or "核心回答")
        aspect = heading.rsplit("/", 1)[-1].strip() or "核心回答"
        grouped.setdefault(aspect, []).append(str(candidate["chunk_id"]))
    if not grouped:
        raise ValueError(f"{case['id']}: no relevance=3 core evidence")
    return [
        {"aspect": aspect, "acceptable_chunk_ids": chunk_ids}
        for aspect, chunk_ids in grouped.items()
    ]


def prepare_dataset(
    candidates_path: Path,
    rerank_path: Path,
) -> dict[str, Any]:
    source = load_json(candidates_path)
    rerank = load_json(rerank_path)
    prepared_cases: list[dict[str, Any]] = []
    for case in source.get("cases", []):
        candidate_map = {str(item["chunk_id"]): item for item in case["candidates"]}
        top_ids = successful_qwen_topk(rerank, str(case["id"]))
        context_sources: list[dict[str, Any]] = []
        seen_context: set[str] = set()
        for chunk_id in top_ids:
            candidate = candidate_map.get(chunk_id) or {}
            for context_source in (candidate.get("context") or {}).get("sources") or []:
                source_id = str(context_source.get("chunk_id") or "")
                if not source_id or source_id == chunk_id or source_id in seen_context:
                    continue
                seen_context.add(source_id)
                context_sources.append(context_source)
        material_ids = list(dict.fromkeys(top_ids + [str(item["chunk_id"]) for item in context_sources]))
        prepared_cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "ground_truth": case.get("ground_truth"),
                "kb_ids": case.get("kb_ids") or [],
                "index_version": source.get("source_index_version", "v2"),
                "baseline_topk_chunk_ids": top_ids,
                "baseline_context_sources": context_sources,
                "baseline_material_chunk_ids": material_ids,
                "candidates": case["candidates"],
                "required_aspects": derive_required_aspects(case),
            }
        )
    annotation = {
        "status": "reviewed",
        "method": "derived_from_existing_document_grounded_human_core_annotations",
        "source_annotation": source.get("annotation"),
        "note": "要点按已人工标为 relevance=3 的章节归并；未使用 baseline 或 evidence 输出改标。",
    }
    dataset = {
        "name": "opt17_evidence_fixed_candidates",
        "created_at": datetime.now(UTC).isoformat(),
        "annotation": annotation,
        "sources": {
            "candidates": {
                "path": str(candidates_path),
                "sha256": sha256_file(candidates_path),
            },
            "rerank_result": {
                "path": str(rerank_path),
                "sha256": sha256_file(rerank_path),
            },
        },
        "cases": prepared_cases,
    }
    dataset["annotation_sha256"] = sha256_json(
        [{"id": case["id"], "required_aspects": case["required_aspects"]} for case in prepared_cases]
    )
    return dataset


def prepare_challenge_five(
    candidates_path: Path,
    rerank_path: Path,
) -> dict[str, Any]:
    dataset = prepare_dataset(candidates_path, rerank_path)
    case_by_id = {str(case["id"]): case for case in dataset["cases"]}
    cases: list[dict[str, Any]] = []
    for case_id, aspects in CHALLENGE_FIVE_ASPECTS.items():
        if case_id not in case_by_id:
            raise ValueError(f"challenge case is missing: {case_id}")
        case = {**case_by_id[case_id], "required_aspects": aspects}
        cases.append(case)
    dataset.update(
        {
            "name": "opt17_evidence_challenge_five",
            "created_at": datetime.now(UTC).isoformat(),
            "annotation": {
                "status": "reviewed",
                "method": "document_grounded_manual_challenge_review",
                "note": (
                    "固定 5 个多要点问题；要点和等价 Chunk 按问题、ground truth "
                    "及原文人工核对，不依据本轮 evidence 输出改标。"
                ),
            },
            "cases": cases,
        }
    )
    dataset["annotation_sha256"] = sha256_json(
        [{"id": case["id"], "required_aspects": case["required_aspects"]} for case in cases]
    )
    return dataset


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        cases = []
    if (dataset.get("annotation") or {}).get("status") != "reviewed":
        errors.append("annotation status must be reviewed")
    for case in cases:
        case_id = str(case.get("id") or "<unknown>")
        candidates = case.get("candidates") or []
        candidate_ids = {str(item.get("chunk_id")) for item in candidates}
        context_ids = {
            str(item.get("chunk_id")) for item in case.get("baseline_context_sources") or []
        }
        top_ids = case.get("baseline_topk_chunk_ids") or []
        if not top_ids or any(str(item) not in candidate_ids for item in top_ids):
            errors.append(f"{case_id}: invalid baseline topk")
        aspects = case.get("required_aspects") or []
        if not aspects:
            errors.append(f"{case_id}: required_aspects is empty")
        seen: set[str] = set()
        for aspect in aspects:
            name = str(aspect.get("aspect") or "").strip()
            acceptable = {str(item) for item in aspect.get("acceptable_chunk_ids") or []}
            if not name or name in seen:
                errors.append(f"{case_id}: empty or duplicate aspect")
            seen.add(name)
            if not acceptable or not acceptable.issubset(candidate_ids | context_ids):
                errors.append(f"{case_id}.{name}: acceptable chunks are invalid")
        for source in case.get("baseline_context_sources") or []:
            if not str(source.get("content") or "").strip():
                errors.append(f"{case_id}: context source has no content")
    expected_hash = sha256_json(
        [{"id": case["id"], "required_aspects": case["required_aspects"]} for case in cases]
    )
    if dataset.get("annotation_sha256") != expected_hash:
        errors.append("annotation SHA-256 does not match")
    return {"valid": not errors, "errors": errors, "case_count": len(cases)}


class FrozenChunkRepository:
    def __init__(self, case: dict[str, Any]) -> None:
        kb_id = str((case.get("kb_ids") or ["eval-kb"])[0])
        index_version = str(case.get("index_version") or "v2")
        sources = list(case["candidates"]) + list(case.get("baseline_context_sources") or [])
        self.chunks = {
            str(item["chunk_id"]): SimpleNamespace(
                id=str(item["chunk_id"]),
                tenant_id="evidence-eval",
                kb_id=str(item.get("kb_id") or kb_id),
                document_id=str(item.get("document_id") or ""),
                title=str(item.get("title") or ""),
                content=str(item.get("content") or ""),
                retrieval_text=item.get("retrieval_text"),
                index_version=str(item.get("index_version") or index_version),
                metadata_=item.get("metadata") or {},
            )
            for item in sources
        }

    def get_by_ids_in_versioned_scope(
        self,
        chunk_ids: list[str],
        *,
        tenant_id: str,
        index_versions: dict[str, str],
    ) -> list[Any]:
        return [
            self.chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.chunks
            and tenant_id == "evidence-eval"
            and index_versions.get(self.chunks[chunk_id].kb_id)
            == self.chunks[chunk_id].index_version
        ]


def build_retrieved_chunks(case: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_map = {str(item["chunk_id"]): item for item in case["candidates"]}
    kb_id = str((case.get("kb_ids") or ["eval-kb"])[0])
    context_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for source in case.get("baseline_context_sources") or []:
        anchor_id = str(source.get("anchor_chunk_id") or "")
        if anchor_id:
            context_by_anchor.setdefault(anchor_id, []).append(source)
    chunks: list[dict[str, Any]] = []
    for chunk_id in case["baseline_topk_chunk_ids"]:
        item = candidate_map[str(chunk_id)]
        sources = [
            {
                "chunk_id": str(chunk_id),
                "relation": "anchor",
                "content": item["content"],
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "metadata": item.get("metadata") or {},
            },
            *context_by_anchor.get(str(chunk_id), []),
        ]
        chunks.append(
            {
                **item,
                "kb_id": item.get("kb_id") or kb_id,
                "index_version": item.get("index_version") or case["index_version"],
                "retrieval_source": item.get("retrieval_source") or "hybrid",
                "context": {
                    "content": "\n\n".join(str(source.get("content") or "") for source in sources),
                    "chunk_ids": [source["chunk_id"] for source in sources],
                    "sources": sources,
                },
            }
        )
    return chunks


async def run_evidence(dataset: dict[str, Any], repeat: int, warmup: int) -> dict[str, Any]:
    if settings.llm_api_key.startswith("your-"):
        raise ValueError("LLM_API_KEY is not configured")
    validation = validate_dataset(dataset)
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed: {validation['errors']}")
    case_results: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        service = EvidenceOrchestrationService(db=SimpleNamespace())
        service.chunk_repository = FrozenChunkRepository(case)
        kb_ids = case.get("kb_ids") or ["eval-kb"]
        index_versions = {str(kb_id): str(case["index_version"]) for kb_id in kb_ids}
        candidates = [
            {
                **item,
                "kb_id": item.get("kb_id") or str(kb_ids[0]),
                "index_version": item.get("index_version") or case["index_version"],
                "retrieval_source": item.get("retrieval_source") or "hybrid",
            }
            for item in case["candidates"]
        ]
        retrieved = build_retrieved_chunks(case)
        runs: list[dict[str, Any]] = []
        for iteration in range(1, warmup + repeat + 1):
            result = await service.orchestrate(
                query=case["question"],
                tenant_id="evidence-eval",
                kb_ids=[str(item) for item in kb_ids],
                index_versions=index_versions,
                candidates=candidates,
                retrieved_chunks=retrieved,
                options=EvidenceOptions(enabled=True, max_items=8),
                rerank_enabled=True,
            )
            usage = getattr(service.llm_provider, "last_usage", {}) or {}
            record = {
                "iteration": iteration,
                "warmup": iteration <= warmup,
                "success": result.pack is not None and not result.metadata.degraded,
                "degraded": result.metadata.degraded,
                "error": result.metadata.error,
                "latency_ms": result.metadata.latency_ms,
                "usage": usage,
                "metadata": result.metadata.model_dump(),
                "evidence_pack": result.pack.model_dump() if result.pack else None,
            }
            runs.append(record)
        case_results.append({"id": case["id"], "question": case["question"], "runs": runs})
    return {
        "name": "opt17_evidence_pack_runs",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset["name"],
        "dataset_annotation_sha256": dataset["annotation_sha256"],
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_sha256": prompt_sha256(),
        "parameters": {
            "repeat": repeat,
            "warmup": warmup,
            "max_items": 8,
            "timeout_seconds": settings.evidence_timeout_seconds,
        },
        "cases": case_results,
    }


def evaluated_ids_from_pack(pack: dict[str, Any] | None) -> set[str]:
    if not pack:
        return set()
    core_ids = {
        reference["evidence_id"]
        for group in pack.get("groups") or []
        for reference in group.get("evidence") or []
        if reference.get("role") == "core"
    }
    return {
        str(item["chunk_id"])
        for item in pack.get("items") or []
        if item.get("evidence_id") in core_ids
    }


def case_metrics(case: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    aspects = case["required_aspects"]
    covered = [
        bool(selected_ids & {str(item) for item in aspect["acceptable_chunk_ids"]})
        for aspect in aspects
    ]
    valid_ids = {
        str(item)
        for aspect in aspects
        for item in aspect["acceptable_chunk_ids"]
    }
    covered_count = sum(covered)
    status = (
        "complete"
        if covered_count == len(aspects)
        else "missing"
        if covered_count == 0
        else "partial"
    )
    return {
        "covered_aspects": covered_count,
        "aspect_count": len(aspects),
        "complete": covered_count == len(aspects),
        "precision": len(selected_ids & valid_ids) / len(selected_ids) if selected_ids else 0.0,
        "status": status,
    }


def summarize(dataset: dict[str, Any], run_result: dict[str, Any]) -> dict[str, Any]:
    dataset_by_id = {str(case["id"]): case for case in dataset["cases"]}
    baseline_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    outside_gain_cases: set[str] = set()
    status_matches = 0
    successful = 0
    degraded = 0
    consistency: list[float] = []
    traced_evidence = 0
    total_evidence = 0
    for case_id, case in dataset_by_id.items():
        baseline_ids = {str(item) for item in case["baseline_material_chunk_ids"]}
        baseline = case_metrics(case, baseline_ids)
        baseline_rows.append({"id": case_id, **baseline, "selected_ids": sorted(baseline_ids)})
        result_case = next(item for item in run_result["cases"] if item["id"] == case_id)
        sets: list[set[str]] = []
        for run in (item for item in result_case["runs"] if not item["warmup"]):
            pack = run["evidence_pack"]
            selected_ids = evaluated_ids_from_pack(pack)
            allowed_ids = {
                str(item["chunk_id"]) for item in case["candidates"]
            } | {
                str(item["chunk_id"]) for item in case.get("baseline_context_sources") or []
            }
            total_evidence += len(selected_ids)
            traced_evidence += len(selected_ids & allowed_ids)
            metrics = case_metrics(case, selected_ids)
            predicted_status = pack.get("status") if pack else None
            status_matches += int(predicted_status == metrics["status"])
            successful += int(run["success"])
            degraded += int(run["degraded"])
            latencies.append(float(run["latency_ms"]))
            usage = run.get("usage") or {}
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key, 0) or 0)
            if not baseline["complete"] and metrics["complete"]:
                outside_gain_cases.add(case_id)
            sets.append(selected_ids)
            evidence_rows.append(
                {
                    "id": case_id,
                    "iteration": run["iteration"],
                    **metrics,
                    "predicted_status": predicted_status,
                    "selected_ids": sorted(selected_ids),
                    "latency_ms": run["latency_ms"],
                    "success": run["success"],
                    "degraded": run["degraded"],
                    "error": run["error"],
                }
            )
        if len(sets) > 1:
            pairs = [
                len(left & right) / len(left | right) if left | right else 1.0
                for index, left in enumerate(sets)
                for right in sets[index + 1 :]
            ]
            consistency.append(statistics.mean(pairs))

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        aspects = sum(row["aspect_count"] for row in rows)
        selected_count = sum(len(row["selected_ids"]) for row in rows)
        candidate_map = {
            str(candidate["chunk_id"]): candidate
            for case in dataset["cases"]
            for candidate in case["candidates"]
        }
        chars = sum(
            len(str(candidate_map.get(chunk_id, {}).get("content") or ""))
            for row in rows
            for chunk_id in row["selected_ids"]
        )
        return {
            "required_aspect_coverage": (
                sum(row["covered_aspects"] for row in rows) / aspects if aspects else 0.0
            ),
            "complete_case_rate": (
                sum(row["complete"] for row in rows) / len(rows) if rows else 0.0
            ),
            "evidence_precision": statistics.mean(row["precision"] for row in rows)
            if rows
            else 0.0,
            "average_evidence_count": selected_count / len(rows) if rows else 0.0,
            "average_evidence_chars": chars / len(rows) if rows else 0.0,
        }

    total_runs = len(evidence_rows)
    return {
        "name": "opt17_evidence_pack_comparison",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": dataset["name"],
            "annotation": dataset["annotation"],
            "annotation_sha256": dataset["annotation_sha256"],
            "case_count": len(dataset["cases"]),
        },
        "runtime": {
            "provider": run_result["provider"],
            "model": run_result["model"],
            "prompt_sha256": run_result["prompt_sha256"],
            "parameters": run_result["parameters"],
        },
        "summary": {
            "baseline_topk_context": aggregate(baseline_rows),
            "evidence_pack": {
                **aggregate(evidence_rows),
                "status_accuracy": status_matches / total_runs if total_runs else 0.0,
                "p50_latency_ms": percentile(latencies, 0.5),
                "p95_latency_ms": percentile(latencies, 0.95),
                "success_rate": successful / total_runs if total_runs else 0.0,
                "degraded_rate": degraded / total_runs if total_runs else 0.0,
                "evidence_set_consistency": statistics.mean(consistency)
                if consistency
                else None,
                "citation_traceability_rate": (
                    traced_evidence / total_evidence if total_evidence else 1.0
                ),
                "usage_totals": usage_totals,
                "average_theoretical_cost": None,
                "cost_note": "未配置统一计价参数，不虚构理论成本。",
            },
            "outside_topk_gain_case_count": len(outside_gain_cases),
            "outside_topk_gain_cases": sorted(outside_gain_cases),
        },
        "baseline_cases": baseline_rows,
        "evidence_runs": evidence_rows,
        "degraded_cases": [row for row in evidence_rows if row["degraded"]],
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["summary"]["baseline_topk_context"]
    evidence = report["summary"]["evidence_pack"]
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    passed = {
        "Evidence Precision ≥ 80%": evidence["evidence_precision"] >= 0.8,
        "平均证据字符数低于 baseline": evidence["average_evidence_chars"]
        < baseline["average_evidence_chars"],
        "引用回溯率 = 100%": evidence["citation_traceability_rate"] == 1.0,
        "5 个 Case 调用成功率 = 100%": evidence["success_rate"] == 1.0,
    }
    lines = [
        "# 优化十七证据包效果测试报告",
        "",
        f"- 测试时间：{report['generated_at']}",
        f"- Case 数：{report['dataset']['case_count']}",
        f"- 标注状态：{report['dataset']['annotation']['status']}",
        f"- Provider / Model：{report['runtime']['provider']} / {report['runtime']['model']}",
        f"- Prompt SHA-256：`{report['runtime']['prompt_sha256']}`",
        f"- 运行参数：repeat={report['runtime']['parameters']['repeat']}，"
        f"warmup={report['runtime']['parameters']['warmup']}，"
        f"timeout={report['runtime']['parameters'].get('timeout_seconds')} 秒",
        "",
        "## 主指标",
        "",
        f"- Required Aspect Coverage：baseline {pct(baseline['required_aspect_coverage'])}；evidence {pct(evidence['required_aspect_coverage'])}",
        f"- Complete Case Rate：baseline {pct(baseline['complete_case_rate'])}；evidence {pct(evidence['complete_case_rate'])}",
        f"- Evidence Precision：baseline {pct(baseline['evidence_precision'])}；evidence {pct(evidence['evidence_precision'])}",
        f"- 平均证据数：baseline {baseline['average_evidence_count']:.2f}；evidence {evidence['average_evidence_count']:.2f}",
        f"- 平均证据字符数：baseline {baseline['average_evidence_chars']:.2f}；evidence {evidence['average_evidence_chars']:.2f}",
        f"- Status Accuracy：{pct(evidence['status_accuracy'])}",
        f"- 引用回溯率：{pct(evidence['citation_traceability_rate'])}",
        f"- TopK 外证据修复 Case：{report['summary']['outside_topk_gain_case_count']}（{', '.join(report['summary']['outside_topk_gain_cases']) or '无'}）",
        "",
        "## 工程指标",
        "",
        f"- P50 / P95：{evidence['p50_latency_ms']:.2f} / {evidence['p95_latency_ms']:.2f} ms",
        f"- 成功率 / 降级率：{pct(evidence['success_rate'])} / {pct(evidence['degraded_rate'])}",
        (
            f"- 证据集合一致率：{pct(evidence['evidence_set_consistency'])}"
            if evidence["evidence_set_consistency"] is not None
            else "- 证据集合一致率：单轮测试，不计算"
        ),
        f"- Token：{evidence['usage_totals']}",
        f"- 成本：{evidence['cost_note']}",
        "",
        "## 验收结果",
        "",
        *[f"- {'通过' if value else '未通过'}：{name}" for name, value in passed.items()],
        "",
        "## 逐题结果",
        "",
    ]
    for row in report["evidence_runs"]:
        lines.append(
            f"- {row['id']} / 第 {row['iteration']} 轮："
            f"coverage {row['covered_aspects']}/{row['aspect_count']}，"
            f"precision {pct(row['precision'])}，status {row['predicted_status']}，"
            f"latency {row['latency_ms']} ms"
        )
    lines.extend(
        [
            "",
            "## 限制与结论",
            "",
            "- 当前源数据未保存 context sources，因此 baseline context 数为 0；报告未虚构补充上下文。",
            f"- 标注方法：{report['dataset']['annotation'].get('method')}；"
            f"{report['dataset']['annotation'].get('note', '')}",
            (
                "- 结论：证据筛选、材料压缩和引用回溯门槛均通过，"
                "证明 evidence_pack 对下游使用有明确价值；本轮未证明 TopK 外缺口修复。"
                if all(passed.values())
                else "- 结论：存在未通过门槛，详见验收结果；原始逐题数据已保留。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--rerank-result", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare_challenge = subparsers.add_parser("prepare-challenge")
    prepare_challenge.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "eval/opt16_context/rerank_comparison.json",
    )
    prepare_challenge.add_argument(
        "--ranking-result",
        dest="rerank_result",
        type=Path,
        default=PROJECT_ROOT / "eval/opt16_context/rerank_llm_vs_qwen3.json",
    )
    prepare_challenge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--dataset", type=Path)
    compare.add_argument("--input", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            save_json(args.output, prepare_dataset(args.candidates, args.rerank_result))
        elif args.command == "prepare-challenge":
            save_json(
                args.output,
                prepare_challenge_five(args.candidates, args.rerank_result),
            )
        elif args.command == "validate":
            result = validate_dataset(load_json(args.dataset))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["valid"] else 1
        elif args.command == "run":
            dataset = load_json(args.dataset)
            save_json(args.output, asyncio.run(run_evidence(dataset, args.repeat, args.warmup)))
        elif args.command == "compare":
            dataset_path = args.dataset or args.input.with_name("evidence_dataset.json")
            report = summarize(load_json(dataset_path), load_json(args.input))
            save_json(args.input, {**load_json(args.input), "comparison": report})
            args.output.write_text(render_markdown(report), encoding="utf-8")
        print(f"saved: {getattr(args, 'output', '')}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
