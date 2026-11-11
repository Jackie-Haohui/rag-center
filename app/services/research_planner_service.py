from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.schemas.research import PlannedQuery, ResearchAspect, normalize_research_query
from app.services.research_limits import MAX_ASPECTS, MAX_INITIAL_QUERIES, MAX_QUERY_CHARS
from app.services.research_query_safety import is_unsafe_retrieval_query

logger = get_logger(__name__)

RESEARCH_PLANNER_PROMPT = """你是 RAG Research 的查询规划器，不负责回答问题。

任务：
1. 先把问题判定为 simple、independent 或 dependent。
2. dependent 表示最终规则依赖必须先从知识库确认的中间条件，例如“岗位名称→职级→差旅等级”或“地点→城市类别→住宿标准”。
3. aspects 必须同时包含回答所需的前置条件和最终答案要点，总数不超过5个。
4. prerequisite_aspect_indexes 只列出首轮必须确认的前置要点；simple 和 independent 必须返回空数组。
5. dependent 的首轮查询只检索前置要点，不提前猜测未知职级、等级、类别或风险值，也不查询依赖这些未知值才能确定的末端规则。
6. simple 和 independent 的首轮查询必须覆盖全部要点；dependent 的首轮查询必须完整覆盖全部前置要点。
7. 生成1～3个首轮查询，每个查询绑定一个或多个要点下标。
8. 多个要点可以共用一个查询，不要为了达到数量而重复改写。
9. 查询必须保留原问题中的关键条件、对象、时间和范围。
10. 不得增加用户没有提出的新目标，不得生成答案，不得选择知识库、索引、租户、网页或外部工具。

严格返回 JSON：
{
  "problem_type": "simple|independent|dependent",
  "aspects": ["简短的必要要点"],
  "prerequisite_aspect_indexes": [0],
  "queries": [{"query": "可独立执行的检索查询", "aspect_indexes": [0]}]
}

dependent 示例：用户询问“研发经理去某城市时的酒店和交通标准”，应把“研发经理对应的职级与差旅等级”以及“目的地城市类别”列为前置要点，首轮只查询这两个前置要点；酒店和交通规则列为答案要点，等待第二轮使用真实检索到的等级和类别查询。不得在首轮查询中猜测研发经理属于哪个等级或目的地属于哪一类。
"""


class PlannerQueryOutput(BaseModel):
    query: str
    aspect_indexes: list[int] = Field(min_length=1)


class PlannerOutput(BaseModel):
    problem_type: Literal["simple", "independent", "dependent"] = "independent"
    aspects: list[str] = Field(min_length=1, max_length=MAX_ASPECTS)
    prerequisite_aspect_indexes: list[int] = Field(default_factory=list)
    queries: list[PlannerQueryOutput] = Field(min_length=1, max_length=MAX_INITIAL_QUERIES)


@dataclass(slots=True)
class PlannerResult:
    aspects: list[ResearchAspect]
    queries: list[PlannedQuery]
    raw_output: dict | None
    model: str
    latency_ms: int
    problem_type: Literal["simple", "independent", "dependent"] = "simple"
    degraded: bool = False
    error: str | None = None
    usage: dict | None = None


class ResearchPlannerService:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or LLMRoleRouter.provider_for(LLMRole.FAST)
        self.model = getattr(self.provider, "model", LLMRoleRouter.model_for(LLMRole.FAST))

    async def plan(self, *, question: str, knowledge_bases: list[Any]) -> PlannerResult:
        started_at = perf_counter()
        logger.info(
            "RESEARCH_PLANNER_START | model=%s | knowledge_base_count=%s",
            self.model,
            len(knowledge_bases),
        )
        raw: dict | None = None
        try:
            raw = await self.provider.chat_json(
                system_prompt=RESEARCH_PLANNER_PROMPT,
                user_payload={
                    "question": question,
                    "knowledge_bases": [
                        {
                            "name": str(getattr(kb, "name", ""))[:255],
                            "description": str(getattr(kb, "description", "") or "")[:500],
                        }
                        for kb in knowledge_bases
                    ],
                    "limits": {
                        "max_aspects": MAX_ASPECTS,
                        "max_queries": MAX_INITIAL_QUERIES,
                        "max_query_chars": MAX_QUERY_CHARS,
                    },
                },
                temperature=0.0,
                timeout_seconds=min(settings.llm_timeout_seconds, 10),
                log_payload=False,
                max_tokens=800,
                thinking="disabled",
            )
            output, aspects, queries = self._validate(raw)
            latency_ms = int((perf_counter() - started_at) * 1000)
            logger.info(
                "RESEARCH_PLANNER_SUCCESS | model=%s | aspect_count=%s | "
                "query_count=%s | latency_ms=%s",
                self.model,
                len(aspects),
                len(queries),
                latency_ms,
            )
            return PlannerResult(
                problem_type=output.problem_type,
                aspects=aspects,
                queries=queries,
                raw_output=raw,
                model=self.model,
                latency_ms=latency_ms,
                usage=self._usage(),
            )
        except Exception as exc:  # noqa: BLE001 - Planner failures must safely fall back.
            latency_ms = int((perf_counter() - started_at) * 1000)
            error = self._safe_error(exc)
            logger.warning(
                "RESEARCH_PLANNER_DEGRADED | model=%s | latency_ms=%s | error=%s",
                self.model,
                latency_ms,
                error,
            )
            aspect = ResearchAspect(aspect_id="A1", description="完整回答用户问题")
            query = PlannedQuery(
                query_id="Q1",
                query=question.strip()[:MAX_QUERY_CHARS],
                aspect_ids=[aspect.aspect_id],
            )
            return PlannerResult(
                problem_type="simple",
                aspects=[aspect],
                queries=[query],
                raw_output=raw,
                model=self.model,
                latency_ms=latency_ms,
                degraded=True,
                error=error,
                usage=self._usage(),
            )

    @classmethod
    def _validate(
        cls,
        raw: dict,
    ) -> tuple[PlannerOutput, list[ResearchAspect], list[PlannedQuery]]:
        try:
            output = PlannerOutput.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("planner output schema is invalid") from exc

        prerequisites = set(output.prerequisite_aspect_indexes)
        if any(index < 0 or index >= len(output.aspects) for index in prerequisites):
            raise ValueError("planner prerequisite aspect index is out of range")
        if output.problem_type == "dependent":
            if not prerequisites or prerequisites == set(range(len(output.aspects))):
                raise ValueError("dependent plan needs premise and answer aspects")
        elif prerequisites:
            raise ValueError("only dependent plans can declare prerequisite aspects")

        normalized_aspects: set[str] = set()
        aspects: list[ResearchAspect] = []
        for index, value in enumerate(output.aspects):
            description = value.strip()
            normalized = normalize_research_query(description)
            if not description or len(description) > 100 or normalized in normalized_aspects:
                raise ValueError("planner aspects contain empty, duplicate, or oversized values")
            normalized_aspects.add(normalized)
            aspects.append(
                ResearchAspect(
                    aspect_id=f"A{index + 1}",
                    description=description,
                    stage="premise" if index in prerequisites else "answer",
                )
            )

        normalized_queries: set[str] = set()
        covered_indexes: set[int] = set()
        queries: list[PlannedQuery] = []
        for index, value in enumerate(output.queries):
            query = value.query.strip()
            normalized = normalize_research_query(query)
            aspect_indexes = value.aspect_indexes
            if not query or len(query) > MAX_QUERY_CHARS or normalized in normalized_queries:
                raise ValueError("planner queries contain empty, duplicate, or oversized values")
            if is_unsafe_retrieval_query(query):
                raise ValueError("planner query is not a safe retrieval query")
            if len(aspect_indexes) != len(set(aspect_indexes)):
                raise ValueError("planner query contains duplicate aspect indexes")
            if any(item < 0 or item >= len(aspects) for item in aspect_indexes):
                raise ValueError("planner aspect index is out of range")
            normalized_queries.add(normalized)
            covered_indexes.update(aspect_indexes)
            queries.append(
                PlannedQuery(
                    query_id=f"Q{index + 1}",
                    query=query,
                    aspect_ids=[aspects[item].aspect_id for item in aspect_indexes],
                )
            )

        required_coverage = (
            prerequisites
            if output.problem_type == "dependent"
            else set(range(len(aspects)))
        )
        if covered_indexes != required_coverage:
            raise ValueError("planner queries do not cover the executable aspects")
        return output, aspects, queries

    def _usage(self) -> dict:
        usage = getattr(self.provider, "last_usage", {})
        return dict(usage) if isinstance(usage, dict) else {}

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, ValueError):
            return str(exc)[:300]
        return type(exc).__name__[:100]
