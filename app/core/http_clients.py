from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from fastapi import Request

from app.core.config import settings


@dataclass(slots=True)
class ExternalHttpClients:
    model: httpx.AsyncClient
    elasticsearch: httpx.AsyncClient

    async def aclose(self) -> None:
        await asyncio.gather(
            self.model.aclose(),
            self.elasticsearch.aclose(),
        )


def build_external_http_clients() -> ExternalHttpClients:
    limits = httpx.Limits(
        max_connections=settings.external_http_max_connections,
        max_keepalive_connections=settings.external_http_max_keepalive_connections,
        keepalive_expiry=settings.external_http_keepalive_expiry_seconds,
    )
    return ExternalHttpClients(
        model=httpx.AsyncClient(limits=limits),
        elasticsearch=httpx.AsyncClient(limits=limits, trust_env=False),
    )


def get_external_http_clients(request: Request) -> ExternalHttpClients:
    clients = getattr(request.app.state, "external_http_clients", None)
    if not isinstance(clients, ExternalHttpClients):
        raise RuntimeError("external HTTP clients are not initialized")
    return clients
