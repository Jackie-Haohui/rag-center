from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.schemas.rag import EvidencePack
from app.schemas.research import (
    PlannedQuery,
    ResearchCandidate,
    ResearchState,
    normalize_research_query,
)
from app.services.evidence_pack_builder import (
    EvidencePackBuilder,
    EvidenceReference,
)
from app.services.research_limits import (
    MAX_CANDIDATE_TEXT_CHARS,
    MAX_LLM_INPUT_CHARS,
    MAX_QUERY_CHARS,
    MAX_SUPPLEMENT_QUERIES,
)
from app.services.research_query_safety import is_unsafe_retrieval_query

logger = get_logger(__name__)

RESEARCH_DECIDER_PROMPT = """你是 RAG Research 的证据判断器，不负责生成最终回答。

任务：
1. 检查每个必要要点是否得到给定候选的直接支持。
2. core 表示候选能够直接回答当前要点。
3. supporting 表示候选只提供必要条件、范围、定义或补充说明。
4. “参见某文档/某章节”“按某规则处理”“进入某流程”只给出了引用路径，不能直接回答规则内容，必须标为 supporting；除非同一候选还明确写出了用户所问的具体条件、步骤、时限或结论。
5. 用户询问“是什么、怎么做、哪些条件、多久”等具体内容时，仅出现主题名或引用目标不算覆盖，该要点仍然缺失。
6. reference_texts 是候选正文提取出的引用线索；补证查询应优先保留其中的准确文档名、章节名，并与缺失要点组合。
7. 主题相似但不能支持结论的候选不能作为证据，不能因为主题匹配就把引用路径标为 core。
8. 每个必要要点必须且只能输出一次；没有证据时返回空 evidence。
9. 同一候选可以支撑多个要点。
10. 如果所有要点都有 core 证据，decision 返回 stop，next_queries 为空。
11. 如果仍有缺失要点，decision 返回 supplement，并为缺失要点生成最多两个定向查询。
12. 补证查询必须使用首轮证据中已经出现的线索，或针对明确缺失要点缩小查询。
13. dependent 问题中，stage=premise 的要点是后续检索必须使用的前置条件。只能把候选正文直接写明的职级、等级、类别、地区组或风险等级带入补证查询。
14. 不得根据岗位名称、城市名称或常识自行猜测中间值；证据没有写明时，补证查询应继续查询缺失的前置条件。
15. 已确认前置条件时，补证查询必须同时保留具体值和末端要点，例如“T2 A类 住宿上限”，不能退化成宽泛的“差旅标准”。
16. 一个补证查询可以覆盖多个相关的缺失末端要点，最多两个查询内应覆盖所有当前可执行的缺失要点。
17. 不得扩大用户问题，不得选择知识库、网页或外部工具。
18. 不生成答案，不改写证据，不输出候选中不存在的事实。

严格返回 JSON：
{
  "aspects": [{"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core|supporting"}]}],
  "decision": "stop|supplement",
  "next_queries": [{"query": "只针对缺失要点的补证查询", "aspect_indexes": [0]}]
}
"""


class StrictOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class DeciderEvidenceOutput(StrictOutput):
    candidate_index: int
    role: Literal["core", "supporting"]


class DeciderAspectOutput(StrictOutput):
    aspect_index: int
    evidence: list[DeciderEvidenceOutput]


class DeciderQueryOutput(StrictOutput):
    query: str
    aspect_indexes: list[int] = Field(min_length=1)


class DeciderOutput(StrictOutput):
    aspects: list[DeciderAspectOutput]
    decision: Literal["stop", "supplement"]
    next_queries: list[DeciderQueryOutput] = Field(
        default_factory=list,
        max_length=MAX_SUPPLEMENT_QUERIES,
    )


@dataclass(slots=True)
class DeciderResult:
    pack: EvidencePack
    selections: dict[str, list[EvidenceReference]]
    next_queries: list[PlannedQuery]
    missing_aspect_ids: list[str]
    raw_output: dict | None
    model: str
    latency_ms: int
    degraded: bool = False
    error: str | None = None
    usage: dict | None = None
    all_next_queries_duplicate: bool = False


class ResearchDeciderService:
    def __init__(
        self,
        pack_builder: EvidencePackBuilder,
        provider: LLMProvider | None = None,
    ) -> None:
        self.pack_builder = pack_builder
        self.provider = provider or LLMRoleRouter.provider_for(LLMRole.FAST)
        self.model = getattr(self.provider, "model", LLMRoleRouter.model_for(LLMRole.FAST))

    async def decide(
        self,
        *,
        state: ResearchState,
        candidates: list[ResearchCandidate],
    ) -> DeciderResult:
        started_at = perf_counter()
        raw: dict | None = None
        try:
            raw = await self.provider.chat_json(
                system_prompt=RESEARCH_DECIDER_PROMPT,
                user_payload=self._payload(state=state, candidates=candidates),
                temperature=0.0,
                timeout_seconds=min(settings.llm_timeout_seconds, 15),
                log_payload=False,
                max_tokens=1200,
                thinking="disabled",
            )
            output = self._validate_output(raw, state=state, candidate_count=len(candidates))
            selections = self._selections(output, state=state)
            build = self.pack_builder.build(
                tenant_id=state.tenant_id,
                index_versions=state.index_versions,
                aspects=state.aspects,
                selections=selections,
                candidates=candidates,
            )
            missing_ids = [
                aspect.aspect_id
                for aspect, group in zip(state.aspects, build.pack.groups, strict=True)
                if not group.covered
            ]
            if build.pack.status == "complete":
                next_queries = []
                all_duplicate = False
            else:
                next_queries, all_duplicate = self._next_queries(
                    output,
                    state=state,
                    missing_ids=missing_ids,
                )
            latency_ms = int((perf_counter() - started_at) * 1000)
            return DeciderResult(
                pack=build.pack,
                selections=selections,
                next_queries=next_queries,
                missing_aspect_ids=missing_ids,
                raw_output=raw,
                model=self.model,
                latency_ms=latency_ms,
                usage=self._usage(),
                all_next_queries_duplicate=all_duplicate,
            )
        except Exception as exc:  # noqa: BLE001 - Decider failures must safely degrade.
            latency_ms = int((perf_counter() - started_at) * 1000)
            error = self._safe_error(exc)
            empty_selections = {aspect.aspect_id: [] for aspect in state.aspects}
            fallback = self.pack_builder.build(
                tenant_id=state.tenant_id,
                index_versions=state.index_versions,
                aspects=state.aspects,
                selections=empty_selections,
                candidates=candidates,
            )
            logger.warning(
                "RESEARCH_DECIDER_DEGRADED | research_id=%s | model=%s | "
                "latency_ms=%s | error=%s",
                state.research_id,
                self.model,
                latency_ms,
                error,
            )
            return DeciderResult(
                pack=fallback.pack,
                selections=empty_selections,
                next_queries=[],
                missing_aspect_ids=[aspect.aspect_id for aspect in state.aspects],
                raw_output=raw,
                model=self.model,
                latency_ms=latency_ms,
                degraded=True,
                error=error,
                usage=self._usage(),
            )

    @staticmethod
    def _validate_output(
        raw: dict,
        *,
        state: ResearchState,
        candidate_count: int,
    ) -> DeciderOutput:
        try:
            output = DeciderOutput.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("decider output schema is invalid") from exc
        if len(output.aspects) != len(state.aspects):
            raise ValueError("decider must return every aspect exactly once")
        aspect_indexes = [item.aspect_index for item in output.aspects]
        if len(set(aspect_indexes)) != len(aspect_indexes) or set(aspect_indexes) != set(
            range(len(state.aspects))
        ):
            raise ValueError("decider aspect indexes are invalid")
        for aspect in output.aspects:
            candidate_indexes = [item.candidate_index for item in aspect.evidence]
            if len(candidate_indexes) != len(set(candidate_indexes)):
                raise ValueError("decider evidence contains duplicate candidate indexes")
            if any(index < 0 or index >= candidate_count for index in candidate_indexes):
                raise ValueError("decider candidate index is out of range")
        return output

    @staticmethod
    def _selections(
        output: DeciderOutput,
        *,
        state: ResearchState,
    ) -> dict[str, list[EvidenceReference]]:
        return {
            state.aspects[item.aspect_index].aspect_id: [
                EvidenceReference(
                    candidate_index=reference.candidate_index,
                    role=reference.role,
                )
                for reference in item.evidence
            ]
            for item in output.aspects
        }

    @staticmethod
    def _next_queries(
        output: DeciderOutput,
        *,
        state: ResearchState,
        missing_ids: list[str],
    ) -> tuple[list[PlannedQuery], bool]:
        missing_indexes = {
            index for index, aspect in enumerate(state.aspects) if aspect.aspect_id in missing_ids
        }
        seen = set(state.visited_queries)
        planned: list[PlannedQuery] = []
        duplicate_count = 0
        query_number = len(state.visited_queries) + 1
        for item in output.next_queries:
            query = item.query.strip()
            normalized = normalize_research_query(query)
            if not query or len(query) > MAX_QUERY_CHARS or is_unsafe_retrieval_query(query):
                raise ValueError("decider supplement query is invalid")
            if normalized in seen:
                duplicate_count += 1
                continue
            if len(item.aspect_indexes) != len(set(item.aspect_indexes)):
                raise ValueError("decider supplement query has duplicate aspect indexes")
            if not set(item.aspect_indexes) or not set(item.aspect_indexes).issubset(
                missing_indexes
            ):
                raise ValueError("decider supplement query must target missing aspects")
            seen.add(normalized)
            planned.append(
                PlannedQuery(
                    query_id=f"Q{query_number}",
                    query=query,
                    aspect_ids=[state.aspects[index].aspect_id for index in item.aspect_indexes],
                )
            )
            query_number += 1
        return planned, bool(output.next_queries) and duplicate_count == len(output.next_queries)

    @staticmethod
    def _payload(
        *,
        state: ResearchState,
        candidates: list[ResearchCandidate],
    ) -> dict:
        payload = {
            "question": state.original_query,
            "problem_type": state.problem_type,
            "aspects": [
                {
                    "index": index,
                    "aspect": aspect.description,
                    "stage": aspect.stage,
                }
                for index, aspect in enumerate(state.aspects)
            ],
            "visited_queries": state.visited_queries,
            "candidates": [
                {
                    "index": index,
                    "title": candidate.title,
                    "heading_path": candidate.heading_path,
                    "text": (candidate.retrieval_text or candidate.content)[
                        :MAX_CANDIDATE_TEXT_CHARS
                    ],
                    "aspect_ids": candidate.aspect_ids,
                    "query_ids": candidate.query_ids,
                    "rounds": candidate.rounds,
                    "source": candidate.source,
                    "context_relation": candidate.context_relation,
                    "reference_texts": list(
                        candidate.metadata.get("reference_texts") or []
                    )[:8],
                }
                for index, candidate in enumerate(candidates)
            ],
            "limits": {
                "max_next_queries": MAX_SUPPLEMENT_QUERIES,
                "max_query_chars": MAX_QUERY_CHARS,
                "remaining_rounds": 1,
            },
        }
        while len(json.dumps(payload, ensure_ascii=False)) > MAX_LLM_INPUT_CHARS:
            longest = max(payload["candidates"], key=lambda item: len(item["text"]), default=None)
            if longest is None or len(longest["text"]) <= 100:
                break
            longest["text"] = longest["text"][:-200]
        return payload

    def _usage(self) -> dict:
        usage = getattr(self.provider, "last_usage", {})
        return dict(usage) if isinstance(usage, dict) else {}

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, (ValueError, PermissionError)):
            return str(exc)[:300]
        return type(exc).__name__[:100]
