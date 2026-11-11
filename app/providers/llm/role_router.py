from __future__ import annotations

from enum import StrEnum

import httpx

from app.core.config import settings
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


class LLMRole(StrEnum):
    FAST = "fast"
    STRONG = "strong"


class LLMRoleRouter:
    @staticmethod
    def model_for(role: LLMRole) -> str:
        if role == LLMRole.STRONG:
            return settings.llm_strong_model or settings.llm_model
        return settings.llm_model

    @classmethod
    def provider_for(
        cls,
        role: LLMRole,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> OpenAICompatibleLLMProvider:
        return OpenAICompatibleLLMProvider(model=cls.model_for(role), client=client)
