from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider
from app.providers.llm.role_router import LLMRole, LLMRoleRouter
from app.schemas.rag import EvidencePack
from app.schemas.research import ResearchCandidate, ResearchState
from app.services.evidence_pack_builder import EvidencePackBuilder, EvidenceReference
from app.services.research_limits import (
    MAX_CANDIDATE_TEXT_CHARS,
    MAX_FINAL_EVIDENCE_ITEMS,
    MAX_LLM_INPUT_CHARS,
)

logger = get_logger(__name__)
MAX_ANSWER_BLOCK_CHARS = 1000
MAX_ANSWER_CHARS = 4000

RESEARCH_FINALIZER_PROMPT = """你是 RAG Research 的最终回答生成器，只能使用给定候选。

任务：
1. 为每个必要要点选择直接支撑结论的 core 证据和必要的 supporting 证据。
2. “参见某文档/某章节”“按某规则处理”“进入某流程”只是引用路径，只能作为 supporting；除非同一候选还写出了用户所问的具体条件、步骤、时限或结论。
3. 用户询问具体规则内容时，不能把主题名或引用目标扩写成答案；只有引用路径而没有目标正文时，应输出 insufficient。
4. reference_texts 仅用于理解候选与被引用规则的关系，本身不是被引用规则的正文。
5. 根据选中的证据生成简洁、自然、可核验的回答块。
6. supported 回答块必须至少引用一条能够直接支持文本的 core 候选。
7. 没有足够证据的要点使用 insufficient，只能说明当前材料不足，且 evidence_indexes 必须为空。
8. 不把常识、模型记忆或外部信息加入回答。
9. 文档存在冲突时明确说明不同来源不一致，并分别引用对应候选。
10. 不自行判断哪个冲突来源更权威。
11. 不输出 Markdown 引用，不输出 Chunk ID，不改写候选正文作为伪造引文。
12. 每个必要要点必须且只能出现在 aspects 中一次。
13. dependent 问题必须保留完整推导链：前置身份/职级、等级/类别等桥接条件和末端规则都要有对应证据。
14. 末端规则虽然命中，但前置条件没有证据时，不得把末端规则套用到用户场景，应输出 insufficient。
15. 回答应直接覆盖用户提出的最终问题；前置要点用于说明推导依据，不要只罗列中间变量而遗漏交通、金额、舱位或审批结论。

严格返回 JSON：
{
  "aspects": [{"aspect_index": 0, "evidence": [{"candidate_index": 0, "role": "core|supporting"}]}],
  "answer_blocks": [{"text": "只由候选支持的回答内容", "evidence_indexes": [0], "kind": "supported|insufficient"}]
}
"""


class StrictOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class FinalizerEvidenceOutput(StrictOutput):
    candidate_index: int
    role: Literal["core", "supporting"]


class FinalizerAspectOutput(StrictOutput):
    aspect_index: int
    evidence: list[FinalizerEvidenceOutput]


class FinalizerAnswerBlock(StrictOutput):
    text: str
    evidence_indexes: list[int]
    kind: Literal["supported", "insufficient"]


class FinalizerOutput(StrictOutput):
    aspects: list[FinalizerAspectOutput]
    answer_blocks: list[FinalizerAnswerBlock]


@dataclass(slots=True)
class FinalizerResult:
    answer: str | None
    pack: EvidencePack
    raw_output: dict | None
    model: str
    latency_ms: int
    degraded: bool = False
    error: str | None = None
    usage: dict | None = None


class ResearchFinalizerService:
    def __init__(
        self,
        pack_builder: EvidencePackBuilder,
        provider: LLMProvider | None = None,
    ) -> None:
        self.pack_builder = pack_builder
        self.provider = provider or LLMRoleRouter.provider_for(LLMRole.STRONG)
        self.model = getattr(self.provider, "model", LLMRoleRouter.model_for(LLMRole.STRONG))

    async def finalize(
        self,
        *,
        state: ResearchState,
        candidates: list[ResearchCandidate],
        fallback_pack: EvidencePack,
    ) -> FinalizerResult:
        started_at = perf_counter()
        raw: dict | None = None
        try:
            raw = await self.provider.chat_json(
                system_prompt=RESEARCH_FINALIZER_PROMPT,
                user_payload=self._payload(state=state, candidates=candidates),
                temperature=0.0,
                timeout_seconds=settings.llm_timeout_seconds,
                log_payload=False,
                max_tokens=1800,
                thinking="disabled",
            )
            output = self._validate_output(raw, state=state, candidates=candidates)
            selections = self._selections(output, state=state)
            evidence_order = [
                index
                for block in output.answer_blocks
                for index in block.evidence_indexes
            ]
            build = self.pack_builder.build(
                tenant_id=state.tenant_id,
                index_versions=state.index_versions,
                aspects=state.aspects,
                selections=selections,
                candidates=candidates,
                evidence_order=evidence_order,
                max_items=MAX_FINAL_EVIDENCE_ITEMS,
            )
            answer = self._render_answer(
                output.answer_blocks,
                evidence_id_by_index=build.evidence_id_by_candidate_index,
            )
            return FinalizerResult(
                answer=answer,
                pack=build.pack,
                raw_output=raw,
                model=self.model,
                latency_ms=int((perf_counter() - started_at) * 1000),
                usage=self._usage(),
            )
        except Exception as exc:  # noqa: BLE001 - Finalizer must preserve safe evidence.
            latency_ms = int((perf_counter() - started_at) * 1000)
            error = self._safe_error(exc)
            logger.warning(
                "RESEARCH_FINALIZER_DEGRADED | research_id=%s | model=%s | "
                "latency_ms=%s | error=%s",
                state.research_id,
                self.model,
                latency_ms,
                error,
            )
            return FinalizerResult(
                answer=None,
                pack=fallback_pack,
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
        candidates: list[ResearchCandidate],
    ) -> FinalizerOutput:
        try:
            output = FinalizerOutput.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("finalizer output schema is invalid") from exc
        if len(output.aspects) != len(state.aspects):
            raise ValueError("finalizer must return every aspect exactly once")
        aspect_indexes = [item.aspect_index for item in output.aspects]
        if len(set(aspect_indexes)) != len(aspect_indexes) or set(aspect_indexes) != set(
            range(len(state.aspects))
        ):
            raise ValueError("finalizer aspect indexes are invalid")
        for aspect in output.aspects:
            indexes = [reference.candidate_index for reference in aspect.evidence]
            if len(indexes) != len(set(indexes)):
                raise ValueError("finalizer evidence contains duplicate indexes")
            if any(index < 0 or index >= len(candidates) for index in indexes):
                raise ValueError("finalizer evidence index is out of range")
        if len(output.answer_blocks) > len(state.aspects) + 2:
            raise ValueError("finalizer returned too many answer blocks")
        total_chars = 0
        referenced: set[int] = set()
        normalized_blocks: list[FinalizerAnswerBlock] = []
        for block in output.answer_blocks:
            text = block.text.strip()
            if not text or len(text) > MAX_ANSWER_BLOCK_CHARS:
                raise ValueError("finalizer answer block text is invalid")
            if re.search(r"\[(?:E|C)\d+\]", text):
                raise ValueError("finalizer answer contains an unresolved reference")
            if any(candidate.chunk_id and candidate.chunk_id in text for candidate in candidates):
                raise ValueError("finalizer answer contains a chunk id")
            indexes = block.evidence_indexes
            if len(indexes) != len(set(indexes)):
                raise ValueError("finalizer answer contains duplicate evidence indexes")
            if any(index < 0 or index >= len(candidates) for index in indexes):
                raise ValueError("finalizer answer evidence index is out of range")
            if block.kind == "supported" and ResearchFinalizerService._is_insufficient_text(text):
                block = block.model_copy(
                    update={
                        "text": "当前材料不足，无法确定。",
                        "evidence_indexes": [],
                        "kind": "insufficient",
                    }
                )
                text = block.text
                indexes = []
            if block.kind == "supported" and not indexes:
                raise ValueError("supported answer block requires evidence")
            if block.kind == "insufficient" and indexes:
                # Some OpenAI-compatible models still echo the candidates they
                # inspected for an explicit “材料不足” conclusion. Those indexes
                # must not become citations, but the otherwise safe no-answer
                # response should not make the whole Finalizer degrade.
                if not ResearchFinalizerService._is_insufficient_text(text):
                    raise ValueError("insufficient answer block cannot assert evidence")
                block = block.model_copy(update={"evidence_indexes": []})
                indexes = []
            total_chars += len(text)
            referenced.update(indexes)
            normalized_blocks.append(block)
        if total_chars > MAX_ANSWER_CHARS:
            raise ValueError("finalizer answer is too long")
        if len(referenced) > MAX_FINAL_EVIDENCE_ITEMS:
            raise ValueError("finalizer selected too many evidence items")
        return output.model_copy(update={"answer_blocks": normalized_blocks})

    @staticmethod
    def _is_insufficient_text(text: str) -> bool:
        normalized = "".join(text.split())
        markers = (
            "材料不足",
            "证据不足",
            "信息不足",
            "未找到",
            "未提及",
            "没有提供",
            "未提供",
            "缺少",
            "无法确定",
            "不能确定",
            "不明确",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _render_answer(
        blocks: list[FinalizerAnswerBlock],
        *,
        evidence_id_by_index: dict[int, str],
    ) -> str:
        rendered: list[str] = []
        for block in blocks:
            references: list[str] = []
            for index in block.evidence_indexes:
                evidence_id = evidence_id_by_index.get(index)
                if evidence_id is None:
                    raise ValueError("finalizer evidence exceeded the safe output budget")
                references.append(f"[{evidence_id}]")
            rendered.append(f"{block.text.strip()}{''.join(references)}")
        return "\n\n".join(rendered)

    @staticmethod
    def _selections(
        output: FinalizerOutput,
        *,
        state: ResearchState,
    ) -> dict[str, list[EvidenceReference]]:
        if output.answer_blocks and all(
            block.kind == "insufficient" for block in output.answer_blocks
        ):
            return {aspect.aspect_id: [] for aspect in state.aspects}
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
