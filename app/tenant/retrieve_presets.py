from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tenant.plan_presets import RetrieveProfileName

FIXED_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrieveProfilePreset:
    """Options bundle for a retrieve profile; consumed by 02 enforcement."""

    top_k: int | None = None
    retrieval_options: dict[str, Any] | None = None
    rerank_options: dict[str, Any] | None = None
    evidence_options: dict[str, Any] | None = None
    query_options: dict[str, Any] | None = None


RETRIEVE_PROFILE_PRESETS: dict[RetrieveProfileName, RetrieveProfilePreset | None] = {
    "speed": RetrieveProfilePreset(
        top_k=5,
        retrieval_options={
            "mode": "vector",
            "vector_top_k": 10,
            "rrf_k": FIXED_RRF_K,
        },
        rerank_options={"enabled": False},
        evidence_options={"enabled": False},
        query_options={"enabled": False, "strategy": "noop", "synonym_enabled": False},
    ),
    "balanced": RetrieveProfilePreset(
        top_k=5,
        retrieval_options={
            "mode": "hybrid",
            "vector_top_k": 10,
            "bm25_top_k": 10,
            "rrf_k": FIXED_RRF_K,
        },
        rerank_options={"enabled": False},
        evidence_options={"enabled": False},
        query_options={"enabled": False, "strategy": "noop", "synonym_enabled": False},
    ),
    "quality": RetrieveProfilePreset(
        top_k=5,
        retrieval_options={
            "mode": "hybrid",
            "vector_top_k": 20,
            "bm25_top_k": 20,
            "rrf_k": FIXED_RRF_K,
        },
        rerank_options={"enabled": True, "top_n": 5},
        evidence_options={"enabled": False},
        query_options={"enabled": True, "strategy": "rewrite", "synonym_enabled": False},
    ),
    "custom": None,
}
