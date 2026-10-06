from unittest.mock import AsyncMock

import httpx
import pytest

from rag_assistant.config import Settings
from rag_assistant.generation import generate
from rag_assistant.store import Hit


@pytest.mark.asyncio
async def test_invalid_citation_from_provider_is_rejected(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        post = AsyncMock()

    response = httpx.Response(
        200,
        json={"message": {"content": '{"answer":"Wrong","citations":["99"]}'}},
        request=httpx.Request("POST", "http://localhost/api/chat"),
    )
    fake = FakeClient()
    fake.post.return_value = response
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: fake)
    hit = Hit("1", "policy.txt", "Annual leave is 20 days.", "public", 0.9)
    with pytest.raises(RuntimeError, match="verifiable citations"):
        await generate("How many days?", [hit], Settings(generator="ollama"))


@pytest.mark.asyncio
async def test_missing_api_key_is_safe_error():
    hit = Hit("1", "policy.txt", "Annual leave is 20 days.", "public", 0.9)
    with pytest.raises(RuntimeError, match="unavailable"):
        await generate("How many days?", [hit], Settings(generator="openai"))
