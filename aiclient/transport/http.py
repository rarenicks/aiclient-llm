import httpx
from typing import Any, Dict, Iterator, AsyncIterator
from .base import Transport

class HTTPTransport(Transport):
    """
    Production-grade HTTP transport using httpx.
    """
    def __init__(self, base_url: str = "", headers: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers
        self.client = httpx.Client(base_url=base_url, headers=headers, timeout=60.0)
        self.aclient = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0)

    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.aclient.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # Use existing client
        with self.client.stream("POST", endpoint, json=data) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    # print(f"DEBUG HTTP STREAM LINE: {line!r}")
                    yield {"raw": line}

    async def stream_async(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        async with self.aclient.stream("POST", endpoint, json=data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    # print(f"DEBUG HTTP ASYNC LINE: {line!r}")
                    yield {"raw": line}
