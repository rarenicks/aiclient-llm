import httpx
from typing import Any, Dict, Iterator, AsyncIterator
from .base import Transport

class HTTPTransport(Transport):
    """
    Production-grade HTTP transport using httpx.
    """
    def __init__(self, base_url: str = "", headers: Dict[str, str] = None, timeout: float = 60.0):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=self.timeout) as client:
            response = client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=self.timeout) as client:
            response = await client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # Simplified streaming implementation
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=self.timeout) as client:
            with client.stream("POST", endpoint, json=data) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield {"raw": line} # Placeholder for actual SSE parsing

    async def stream_async(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
         async with httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=self.timeout) as client:
            async with client.stream("POST", endpoint, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield {"raw": line}
