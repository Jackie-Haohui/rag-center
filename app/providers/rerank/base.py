from abc import ABC, abstractmethod


class RerankProviderError(RuntimeError):
    """Raised when a rerank provider cannot return a trustworthy ranking."""


class RerankProvider(ABC):
    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        pass
