#!/usr/bin/env python3
"""Export low-score retrieval feedback from Langfuse into eval dataset candidates."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import (  # noqa: E402
    FEEDBACK_SCORE_NAME,
    extract_trace_metadata,
    extract_trace_question,
    get_langfuse_for_scripts,
    load_dataset,
    merge_cases,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export user_feedback scores below a threshold from Langfuse.",
    )
    parser.add_argument("--max-score", type=float, default=3.0, help="Export scores < this value")
    parser.add_argument("--days", type=int, default=30, help="Look back N days")
    parser.add_argument("--kb-id", default=None, help="Only include traces for this kb_id")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/imported_from_feedback.json"),
        help="Output dataset JSON path",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="Merge exported cases into an existing dataset (dedupe by question)",
    )
    parser.add_argument("--name", default="imported_from_feedback", help="Dataset name")
    return parser.parse_args()


def fetch_low_scores(langfuse: Any, *, max_score: float, days: int) -> list[Any]:
    from_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
    page = 1
    scores: list[Any] = []

    while True:
        response = langfuse.api.score.get(
            name=FEEDBACK_SCORE_NAME,
            operator="<",
            value=max_score,
            from_timestamp=from_timestamp,
            limit=100,
            page=page,
        )
        batch = response.data or []
        if not batch:
            break
        scores.extend(batch)
        meta = response.meta
        total_pages = getattr(meta, "total_pages", None) or getattr(meta, "totalPages", None)
        if total_pages is not None and page >= total_pages:
            break
        if len(batch) < 100:
            break
        page += 1

    return scores


def build_case_from_score(langfuse: Any, score: Any, *, kb_filter: str | None) -> dict[str, Any] | None:
    trace_id = getattr(score, "trace_id", None)
    if not trace_id:
        print(f"WARN: skip score {getattr(score, 'id', None)}: missing trace_id", file=sys.stderr)
        return None

    try:
        fetched = langfuse.fetch_trace(trace_id)
    except Exception as exc:
        print(f"WARN: skip trace {trace_id}: fetch failed: {exc}", file=sys.stderr)
        return None

    trace = fetched.data
    
    # 检查 1: Trace 名字对不对
    trace_name = getattr(trace, "name", None)
    if trace_name != "rag_research":
        print(f"WARN: skip trace {trace_id}: trace name is '{trace_name}', expected 'rag_research'", file=sys.stderr)
        return None

    # 检查 2: 能不能提取到问题 (query)
    trace_input = getattr(trace, "input", None)
    question = extract_trace_question(trace_input)
    if not question:
        print(f"WARN: skip trace {trace_id}: missing query in trace input. Raw input: {trace_input}", file=sys.stderr)
        return None

    # 检查 3: kb_id 过滤（已修复：兼容 index_versions 嵌套字典的情况）
    metadata = extract_trace_metadata(trace)
    kb_ids = metadata.get("kb_ids")
    if isinstance(kb_ids, list):
        normalized_kb_ids = [str(item).strip() for item in kb_ids if str(item).strip()]
    else:
        normalized_kb_ids = []

    # 【核心修复】从 index_versions 字典的 Key 中提取 kb_id 兜底
    index_versions = metadata.get("index_versions")
    if isinstance(index_versions, dict):
        normalized_kb_ids.extend([str(k).strip() for k in index_versions.keys() if str(k).strip()])
    
    # 去重，防止重复
    normalized_kb_ids = list(set(normalized_kb_ids))
        
    kb_id = normalized_kb_ids[0] if normalized_kb_ids else metadata.get("kb_id")
    
    if kb_filter and kb_id != kb_filter and kb_filter not in normalized_kb_ids:
        print(f"WARN: skip trace {trace_id}: kb_id mismatch. Expected '{kb_filter}', got kb_id='{kb_id}', kb_ids={normalized_kb_ids}, metadata={metadata}", file=sys.stderr)
        return None

    short_id = trace_id.replace("-", "")[:8]
    source: dict[str, Any] = {
        "trace_id": trace_id,
        "log_id": metadata.get("log_id"),
        "feedback_score": int(getattr(score, "value", 0)),
        "feedback_comment": getattr(score, "comment", None),
        "kb_id": kb_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    if normalized_kb_ids:
        source["kb_ids"] = normalized_kb_ids
    return {
        "id": f"lf_{short_id}",
        "question": question,
        "ground_truth": "",
        "source": source,
    }


def main() -> int:
    args = parse_args()

    try:
        langfuse = get_langfuse_for_scripts()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    scores = fetch_low_scores(langfuse, max_score=args.max_score, days=args.days)
    if not scores:
        print(f"未找到 {FEEDBACK_SCORE_NAME} 分数 < {args.max_score} 的记录（近 {args.days} 天）")
        return 0

    exported_cases: list[dict[str, Any]] = []
    kb_ids: set[str] = set()
    for score in scores:
        case = build_case_from_score(langfuse, score, kb_filter=args.kb_id)
        if case is None:
            continue
        exported_cases.append(case)
        source_kb = case.get("source", {}).get("kb_id")
        if isinstance(source_kb, str) and source_kb:
            kb_ids.add(source_kb)

    if not exported_cases:
        print("没有匹配 rag_research trace 的低分记录（请查看上方 WARN 日志排查原因）")
        return 0

    kb_id = args.kb_id or (next(iter(kb_ids)) if len(kb_ids) == 1 else "")
    if len(kb_ids) > 1 and not args.kb_id:
        print(
            f"WARN: exported traces span multiple kb_id values {sorted(kb_ids)}; "
            "set --kb-id or edit output JSON before running eval",
            file=sys.stderr,
        )

    dataset = {
        "name": args.name,
        "kb_id": kb_id,
        "default_profile": "balanced",
        "cases": exported_cases,
    }

    if args.merge:
        try:
            target = load_dataset(args.merge)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        merged_cases, added = merge_cases(target.get("cases", []), exported_cases)
        target["cases"] = merged_cases
        if kb_id and not target.get("kb_id"):
            target["kb_id"] = kb_id
        dataset = target
        output_path = args.merge if args.output == Path("eval/datasets/imported_from_feedback.json") else args.output
        save_dataset(output_path, dataset)
        pending = sum(1 for case in merged_cases if not str(case.get("ground_truth") or "").strip())
        print(f"已合并 {added} 条新测例 → {output_path}")
        print(f"当前共 {len(merged_cases)} 条 · 待补 ground_truth（标准答案）：{pending} 条")
        return 0

    save_dataset(args.output, dataset)
    pending = len(exported_cases)
    print(f"已导出 {len(exported_cases)} 条低分测例 → {args.output}")
    print(f"待补 ground_truth（标准答案）：{pending} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())