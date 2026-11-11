import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import httpx

from app.core.config import settings
from app.core.logging import log_llm_error, log_llm_request, log_llm_response
from app.providers.llm.base import LLMProvider, LLMProviderError


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        base_url: str = settings.llm_base_url,
        api_key: str = settings.llm_api_key,
        model: str = settings.llm_model,
        timeout_seconds: int = settings.llm_timeout_seconds,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = client
        self.last_usage: dict = {}

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            yield client

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        temperature: float = 0.0,
        timeout_seconds: int | None = None,
        log_payload: bool = True,
        max_tokens: int | None = None,
        thinking: Literal["enabled", "disabled"] | None = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started_at = time.perf_counter()
        candidates = user_payload.get("candidates")
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        input_chars = len(json.dumps(user_payload, ensure_ascii=False))
        logged_prompt = (
            {"system_prompt": system_prompt, "user_payload": user_payload}
            if log_payload
            else {"payload_redacted": True}
        )
        log_llm_request(
            model=self.model,
            prompt=logged_prompt,
            extra={
                "provider": "openai_compatible",
                "base_url": self.base_url,
                "candidate_count": candidate_count,
                "input_chars": input_chars,
            },
        )

        try:
            request_timeout = timeout_seconds or self.timeout_seconds
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            cost_ms = (time.perf_counter() - started_at) * 1000
            log_llm_error(
                model=self.model,
                error=exc,
                cost_ms=cost_ms,
                prompt=logged_prompt,
                extra={
                    "provider": "openai_compatible",
                    "base_url": self.base_url,
                    "candidate_count": candidate_count,
                    "input_chars": input_chars,
                },
            )
            raise LLMProviderError(f"llm request failed: {exc}") from exc

        try:
            body = response.json()
            usage = body.get("usage") if isinstance(body, dict) else None
            self.last_usage = usage if isinstance(usage, dict) else {}
            content = body["choices"][0]["message"]["content"]
            parsed = self._parse_json_content(content)
            cost_ms = (time.perf_counter() - started_at) * 1000
            log_llm_response(
                model=self.model,
                response=parsed if log_payload else {"payload_redacted": True},
                cost_ms=cost_ms,
                extra={
                    "provider": "openai_compatible",
                    "candidate_count": candidate_count,
                    "input_chars": input_chars,
                },
            )
            return parsed
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            cost_ms = (time.perf_counter() - started_at) * 1000
            log_llm_error(
                model=self.model,
                error=exc,
                cost_ms=cost_ms,
                prompt=logged_prompt,
                extra={
                    "provider": "openai_compatible",
                    "candidate_count": candidate_count,
                    "input_chars": input_chars,
                },
            )
            raise LLMProviderError(f"llm response json parse failed: {exc}") from exc

    @staticmethod
    def _parse_json_content(content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("top-level JSON is not an object", cleaned, 0)
        return parsed
