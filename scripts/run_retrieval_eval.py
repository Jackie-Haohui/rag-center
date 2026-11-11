#!/usr/bin/env python3
"""Run offline retrieval evaluation with RAGAS context_precision / context_recall."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import httpx
from datasets import Dataset
from ragas import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import (  # noqa: E402
    EVAL_USER_ID,
    METRIC_HINTS,
    METRIC_LABELS,
    PROFILE_HINTS,
    build_ragas_llm,
    ground_truth_ready,
    load_dataset,
    load_ragas_metrics,
)

LOW_RECALL_THRESHOLD = 0.5
LOW_PRECISION_THRESHOLD = 0.5
LINE = "═" * 56


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS retrieval evaluation on a golden set.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSON")
    parser.add_argument("--api-key", default=None, help="RAG API key (Bearer rk_live_...)")
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="本地评测租户；未提供 API Key 时创建并自动删除临时 Key",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="RAG Center base URL")
    parser.add_argument("--profile", default=None, help="Retrieve profile override")
    parser.add_argument(
        "--request-config",
        type=Path,
        default=None,
        help="合并进每次 retrieve 请求的冻结配置 JSON",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report to this path")
    parser.add_argument(
        "--low-recall-threshold",
        type=float,
        default=LOW_RECALL_THRESHOLD,
        help="List cases with context_recall below this value",
    )
    parser.add_argument(
        "--low-precision-threshold",
        type=float,
        default=LOW_PRECISION_THRESHOLD,
        help="List cases with context_precision below this value",
    )
    return parser.parse_args()


def retrieve_case(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    kb_ids: list[str],
    question: str,
    profile: str,
    request_config: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": EVAL_USER_ID,
        "query": question,
        "profile": profile,
        **request_config,
    }
    if len(kb_ids) == 1:
        payload["kb_id"] = kb_ids[0]
    else:
        payload["kb_ids"] = kb_ids

    response = client.post(
        f"{base_url.rstrip('/')}/api/v1/rag/retrieve",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    body = response.json()
    code = body.get("code")
    if code != 0:
        msg = body.get("msg") or "retrieve failed"
        raise RuntimeError(f"retrieve error code={code}: {msg}")
    return body["data"]


def resolve_case_kb_ids(case: dict[str, Any], default_kb_id: str) -> list[str]:
    raw_ids = case.get("kb_ids")
    if isinstance(raw_ids, list):
        kb_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        if kb_ids:
            return kb_ids
    kb_id = str(case.get("kb_id") or default_kb_id).strip()
    return [kb_id] if kb_id else []


def collect_eval_rows(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    default_kb_id: str,
    profile: str,
    request_config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    print("\n【第 1 步】批量检索（调用 /rag/retrieve）")
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or case.get("question") or "case")
        question = str(case.get("question") or "").strip()
        reference = str(case.get("ground_truth") or "").strip()
        case_kb_ids = resolve_case_kb_ids(case, default_kb_id)
        try:
            started_at = perf_counter()
            data = retrieve_case(
                client,
                base_url=base_url,
                api_key=api_key,
                kb_ids=case_kb_ids,
                question=question,
                profile=profile,
                request_config=request_config,
            )
            latency_ms = (perf_counter() - started_at) * 1000
        except Exception as exc:
            failures.append({"id": case_id, "question": question, "error": str(exc)})
            print(f"  ✗ [{index}/{len(cases)}] {case_id}")
            print(f"    问题：{question}")
            print(f"    原因：{exc}")
            continue

        chunks = data.get("retrieved_chunks") or []
        contexts = [str(chunk.get("content") or "") for chunk in chunks if chunk.get("content")]
        rows.append(
            {
                "id": case_id,
                "question": question,
                "reference": reference,
                "user_input": question,
                "retrieved_contexts": contexts,
                "multi_kb": len(data.get("kb_ids") or case_kb_ids) > 1,
                "kb_ids": data.get("kb_ids") or case_kb_ids,
                "latency_ms": round(latency_ms, 3),
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "kb_id": chunk.get("kb_id"),
                        "document_id": chunk.get("document_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "title": chunk.get("title"),
                        "content": chunk.get("content"),
                        "score": chunk.get("score"),
                        "vector_score": chunk.get("vector_score"),
                        "bm25_score": chunk.get("bm25_score"),
                        "retrieval_source": chunk.get("retrieval_source"),
                        "index_version": chunk.get("index_version"),
                    }
                    for rank, chunk in enumerate(chunks, start=1)
                ],
                "metadata": data.get("metadata") or {},
            }
        )
        multi_label = "multi_kb" if len(data.get("kb_ids") or case_kb_ids) > 1 else "single_kb"
        print(f"  ✓ [{index}/{len(cases)}] {case_id} — 召回 {len(contexts)} 条 chunk ({multi_label})")

    return rows, failures


def summarize_scores(df, metric_names: list[str]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for name in metric_names:
        if name not in df.columns:
            summary[name] = None
            continue
        series = df[name].dropna()
        summary[name] = float(series.mean()) if not series.empty else None
    return summary


def format_percent(value: float | None) -> str:
    if value is None:
        return "无分数"
    return f"{value * 100:.1f}%"


def score_level(value: float | None) -> str:
    if value is None:
        return "未评分"
    if value >= 0.85:
        return "优秀"
    if value >= 0.70:
        return "良好"
    if value >= 0.50:
        return "一般"
    return "较差"


def build_metric_readable(metric_key: str, value: float | None) -> dict[str, str]:
    label = METRIC_LABELS.get(metric_key, metric_key)
    return {
        "metric": metric_key,
        "label": label,
        "hint": METRIC_HINTS.get(metric_key, ""),
        "percent": format_percent(value),
        "level": score_level(value),
    }


def build_case_readable(item: dict[str, Any]) -> dict[str, Any]:
    precision = item.get("context_precision")
    recall = item.get("context_recall")
    scores = [value for value in (recall, precision) if value is not None]
    worst = min(scores) if scores else None
    return {
        "id": item["id"],
        "question": item["question"],
        "上下文精确率": format_percent(precision),
        "上下文召回率": format_percent(recall),
        "综合评价": score_level(worst),
    }


def print_header(title: str) -> None:
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def print_report_intro(
    *,
    dataset_path: Path,
    kb_id: str,
    profile: str,
    total: int,
    runnable: int,
    skipped: int,
) -> None:
    print_header("检索离线评测")
    print(f"  测例文件   {dataset_path}")
    print(f"  知识库 ID  {kb_id}")
    print(f"  检索策略   {profile} — {PROFILE_HINTS.get(profile, '见后端 preset')}")
    print(f"  测例统计   共 {total} 条 · 参与评测 {runnable} 条 · 跳过 {skipped} 条")
    if skipped:
        print("  （跳过的测例 ground_truth 为空，需先补标准答案）")


def print_summary_readable(summary: dict[str, float | None]) -> None:
    print_header("汇总得分（0%～100%，越高越好）")
    for key in ("context_recall", "context_precision"):
        readable = build_metric_readable(key, summary.get(key))
        print(f"  {readable['label']}  {readable['percent']}  [{readable['level']}]")
        print(f"    └ {readable['hint']}")


def case_passes(item: dict[str, Any]) -> bool:
    recall = item.get("context_recall")
    precision = item.get("context_precision")
    if recall is not None and recall < 0.85:
        return False
    if precision is not None and precision < 0.85:
        return False
    return recall is not None or precision is not None


def print_per_case_readable(per_case: list[dict[str, Any]]) -> None:
    print_header("逐条明细")
    for index, item in enumerate(per_case, start=1):
        readable = build_case_readable(item)
        mark = "✓" if case_passes(item) else "!"
        recall = item.get("context_recall")
        precision = item.get("context_precision")
        print(f"  {mark} {index}. {item['id']}")
        print(f"    问题：{item['question']}")
        print(
            f"    召回率 {format_percent(recall)}  "
            f"精确率 {format_percent(precision)}  "
            f"→ {readable['综合评价']}"
        )


def print_attention_readable(
    *,
    low_recall: list[dict[str, Any]],
    low_precision: list[dict[str, Any]],
    recall_threshold: float,
    precision_threshold: float,
) -> None:
    if not low_recall and not low_precision:
        print(
            "\n【需关注】无低分测例（召回率、精确率均 ≥ {:.0f}%）".format(
                recall_threshold * 100
            )
        )
        return

    print_header("需关注")
    if low_recall:
        print(f"  召回率低于 {recall_threshold * 100:.0f}% 的测例：")
        for item in low_recall:
            print(f"  ! {item['id']}")
            print(f"    问题：{item['question']}")
            print(f"    召回率：{format_percent(item.get('context_recall'))}")
    if low_precision:
        if low_recall:
            print()
        print(f"  精确率低于 {precision_threshold * 100:.0f}% 的测例：")
        for item in low_precision:
            print(f"  ! {item['id']}")
            print(f"    问题：{item['question']}")
            print(f"    精确率：{format_percent(item.get('context_precision'))}")


def print_retrieve_failures_readable(failures: list[dict[str, str]]) -> None:
    if not failures:
        return
    print_header("检索失败")
    for item in failures:
        print(f"  ✗ {item['id']}")
        print(f"    问题：{item['question']}")
        print(f"    原因：{item['error']}")


def build_report_payload(
    *,
    args: argparse.Namespace,
    kb_id: str,
    profile: str,
    summary: dict[str, float | None],
    per_case: list[dict[str, Any]],
    low_recall: list[dict[str, Any]],
    low_precision: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    retrieve_failures: list[dict[str, str]],
    rows: list[dict[str, Any]],
    request_config: dict[str, Any],
) -> dict[str, Any]:
    summary_readable = {
        METRIC_LABELS["context_recall"]: build_metric_readable(
            "context_recall", summary.get("context_recall")
        ),
        METRIC_LABELS["context_precision"]: build_metric_readable(
            "context_precision", summary.get("context_precision")
        ),
        "参与评测": len(per_case),
        "跳过测例": len(skipped),
        "检索失败": len(retrieve_failures),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "kb_id": kb_id,
        "profile": profile,
        "request_config": request_config,
        "profile_description": PROFILE_HINTS.get(profile, ""),
        "summary": {
            **summary,
            "evaluated_cases": len(per_case),
            "skipped_cases": len(skipped),
            "retrieve_failures": len(retrieve_failures),
        },
        "summary_readable": summary_readable,
        "per_case": per_case,
        "per_case_readable": [build_case_readable(item) for item in per_case],
        "low_recall_cases": low_recall,
        "low_precision_cases": low_precision,
        "retrieve_failures": retrieve_failures,
        "performance_by_kb_count": summarize_performance_by_kb_count(rows),
        "retrieval_rows": rows,
    }


def summarize_performance_by_kb_count(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        kb_count = len(row.get("kb_ids") or [])
        grouped.setdefault(kb_count, []).append(float(row["latency_ms"]))
    return {
        str(kb_count): {
            "case_count": len(values),
            "latency_mean_ms": mean(values) if values else None,
            "latency_p50_ms": median(values) if values else None,
            "latency_p95_ms": _percentile(values, 0.95),
            "latency_max_ms": max(values) if values else None,
        }
        for kb_count, values in sorted(grouped.items())
    }


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


def main() -> int:
    args = parse_args()

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    kb_id = str(dataset.get("kb_id") or "").strip()
    if not kb_id or kb_id == "REPLACE_WITH_YOUR_KB_ID":
        print("错误：dataset 里的 kb_id 未填写或仍是占位符 REPLACE_WITH_YOUR_KB_ID", file=sys.stderr)
        return 1

    profile = args.profile or str(dataset.get("default_profile") or "balanced")
    try:
        request_config = load_request_config(args.request_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    if request_config.get("profile"):
        profile = str(request_config["profile"])
    all_cases = dataset.get("cases", [])
    runnable = [case for case in all_cases if ground_truth_ready(case)]
    skipped = [case for case in all_cases if not ground_truth_ready(case)]

    print_report_intro(
        dataset_path=args.dataset,
        kb_id=kb_id,
        profile=profile,
        total=len(all_cases),
        runnable=len(runnable),
        skipped=len(skipped),
    )
    if skipped:
        print("\n  跳过的测例：")
        for case in skipped:
            print(f"    - {case.get('id') or case.get('question')}")

    if not runnable:
        print("\n错误：没有可评测测例，请先为 case 填写 ground_truth（标准答案）", file=sys.stderr)
        return 1

    try:
        llm = build_ragas_llm()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    context_precision, context_recall = load_ragas_metrics()
    metric_names = ["context_precision", "context_recall"]

    try:
        with evaluation_api_key(args) as api_key:
            with httpx.Client(timeout=120.0, trust_env=False) as client:
                rows, retrieve_failures = collect_eval_rows(
                    client,
                    base_url=args.base_url,
                    api_key=api_key,
                    default_kb_id=kb_id,
                    profile=profile,
                    request_config=request_config,
                    cases=runnable,
                )
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if not rows:
        print("\n错误：所有 retrieve 调用均失败", file=sys.stderr)
        return 1

    eval_dataset = Dataset.from_dict(
        {
            "user_input": [row["user_input"] for row in rows],
            "reference": [row["reference"] for row in rows],
            "retrieved_contexts": [row["retrieved_contexts"] for row in rows],
        }
    )

    print("\n【第 2 步】RAGAS 评分（LLM 评判召回质量，约需数分钟…）")
    result = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        show_progress=True,
        raise_exceptions=False,
    )
    scores_df = result.to_pandas()

    per_case: list[dict[str, Any]] = []
    low_recall: list[dict[str, Any]] = []
    low_precision: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        case_scores = scores_df.iloc[index]
        item = {
            "id": row["id"],
            "question": row["question"],
            "multi_kb": row.get("multi_kb", False),
            "kb_ids": row.get("kb_ids") or [],
            "context_precision": _maybe_float(case_scores.get("context_precision")),
            "context_recall": _maybe_float(case_scores.get("context_recall")),
        }
        per_case.append(item)
        recall = item["context_recall"]
        precision = item["context_precision"]
        if recall is not None and recall < args.low_recall_threshold:
            low_recall.append(item)
        if precision is not None and precision < args.low_precision_threshold:
            low_precision.append(item)

    summary = summarize_scores(scores_df, metric_names)
    print_summary_readable(summary)
    print_per_case_readable(per_case)
    print_attention_readable(
        low_recall=low_recall,
        low_precision=low_precision,
        recall_threshold=args.low_recall_threshold,
        precision_threshold=args.low_precision_threshold,
    )
    print_retrieve_failures_readable(retrieve_failures)

    if args.output:
        report = build_report_payload(
            args=args,
            kb_id=kb_id,
            profile=profile,
            summary=summary,
            per_case=per_case,
            low_recall=low_recall,
            low_precision=low_precision,
            skipped=skipped,
            retrieve_failures=retrieve_failures,
            rows=rows,
            request_config=request_config,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\n完整报告已写入：{args.output}")
        print("  （JSON 中含 summary_readable / per_case_readable 中文字段）")

    print(f"\n{LINE}\n")
    return 0 if not retrieve_failures else 2


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        return None
    return float(value)


def load_request_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"冻结请求配置不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"冻结请求配置 JSON 无效: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("冻结请求配置必须是 JSON 对象")
    forbidden = {"query", "user_id", "kb_id", "kb_ids"} & payload.keys()
    if forbidden:
        raise ValueError("冻结请求配置不能包含: " + ", ".join(sorted(forbidden)))
    return payload


@contextmanager
def evaluation_api_key(args: argparse.Namespace) -> Iterator[str]:
    if args.api_key:
        yield args.api_key
        return
    if not args.tenant_id:
        raise RuntimeError("缺少评测身份：请传 --api-key 或 --tenant-id")

    from app.db.session import SessionLocal
    from app.models.api_key import ApiKey
    from app.repositories.api_key_repository import ApiKeyRepository
    from app.repositories.tenant_repository import TenantRepository
    from app.utils.api_key_crypto import generate_api_key_material
    from app.utils.id_generator import generate_uuid
    from app.utils.tenant_status import ApiKeyStatus

    db = SessionLocal()
    key_id = generate_uuid()
    key_created = False
    try:
        tenant = TenantRepository(db).get_by_id(args.tenant_id)
        if tenant is None:
            raise RuntimeError(f"评测租户不存在: {args.tenant_id}")
        raw_key, key_hash, key_prefix = generate_api_key_material()
        ApiKeyRepository(db).create(
            ApiKey(
                id=key_id,
                tenant_id=args.tenant_id,
                name="retrieval-eval-temporary",
                key_prefix=key_prefix,
                key_hash=key_hash,
                status=ApiKeyStatus.ACTIVE,
                expires_at=None,
            )
        )
        db.commit()
        key_created = True
        yield raw_key
    finally:
        try:
            if key_created:
                api_key = db.get(ApiKey, key_id)
                if api_key is not None:
                    db.delete(api_key)
                    db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
