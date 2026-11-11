from __future__ import annotations

from collections import Counter
from typing import Any

from app.utils.markdown_splitter import is_low_information_content, normalize_content


def filter_retrieval_candidates(chunks: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    filtered: list[dict] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    reasons: Counter[str] = Counter()

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        metadata = chunk.get("metadata") or {}
        if metadata.get("chunk_type") == "reference_pointer":
            reasons["reference_pointer"] += 1
            continue
        heading_path = str(metadata.get("heading_path") or "")
        heading = heading_path.rsplit("/", 1)[-1] if heading_path else None
        content = str(chunk.get("content") or "")
        if is_low_information_content(content, heading):
            reasons["low_information"] += 1
            continue
        if chunk_id and chunk_id in seen_ids:
            reasons["duplicate_chunk_id"] += 1
            continue
        normalized = normalize_content(content)
        if normalized in seen_content:
            reasons["duplicate_content"] += 1
            continue
        if chunk_id:
            seen_ids.add(chunk_id)
        seen_content.add(normalized)
        filtered.append(chunk)

    return filtered, {
        "candidate_count_before_filter": len(chunks),
        "candidate_count_after_filter": len(filtered),
        "filtered_count": len(chunks) - len(filtered),
        "filtered_reasons": dict(reasons),
    }
