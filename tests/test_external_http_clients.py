import asyncio

import httpx

from app.core.http_clients import ExternalHttpClients
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.keyword_search.elasticsearch import ElasticsearchKeywordSearchProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.rerank.qwen3 import Qwen3RerankProvider


def test_online_providers_use_injected_clients_without_closing_them():
    requests: list[str] = []

    def model_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            )
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.9}]},
        )

    def elasticsearch_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, json={"hits": {"hits": []}})

    async def scenario():
        model_client = httpx.AsyncClient(transport=httpx.MockTransport(model_handler))
        elasticsearch_client = httpx.AsyncClient(
            transport=httpx.MockTransport(elasticsearch_handler),
            trust_env=False,
        )
        clients = ExternalHttpClients(model=model_client, elasticsearch=elasticsearch_client)
        embedding = OpenAICompatibleEmbeddingProvider(
            base_url="https://models.example/v1",
            api_key="key",
            dimensions=1,
            client=model_client,
        )
        rerank = Qwen3RerankProvider(
            base_url="https://models.example/v1",
            api_key="key",
            client=model_client,
        )
        llm = OpenAICompatibleLLMProvider(
            base_url="https://models.example/v1",
            api_key="key",
            client=model_client,
        )
        keyword = ElasticsearchKeywordSearchProvider(
            base_url="https://es.example",
            client=elasticsearch_client,
        )

        await embedding.embed_query("query")
        await llm.chat_json(system_prompt="system", user_payload={"query": "query"})
        await rerank.rerank(
            query="query",
            chunks=[{"chunk_id": "c1", "title": "t", "content": "c"}],
            top_n=1,
        )
        await keyword.keyword_search_scope(
            query="query",
            tenant_id="tenant-a",
            index_versions={"kb-a": "v2"},
            top_k=10,
        )

        assert not model_client.is_closed
        assert not elasticsearch_client.is_closed
        await clients.aclose()
        assert model_client.is_closed
        assert elasticsearch_client.is_closed

    asyncio.run(scenario())
    assert requests == [
        "/v1/embeddings",
        "/v1/chat/completions",
        "/v1/reranks",
        "/rag_chunks/_search",
    ]
