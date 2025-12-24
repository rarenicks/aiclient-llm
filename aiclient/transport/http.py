import logging
import httpx
from typing import Any, Dict, Iterator, AsyncIterator
from .base import Transport
from ..exceptions import (
    AuthenticationError, 
    RateLimitError, 
    ProviderError, 
    InvalidRequestError, 
    NetworkError, 
    AIClientError
)

logger = logging.getLogger("aiclient.transport")

class HTTPTransport(Transport):
    """
    Production-grade HTTP transport using httpx.
    """
    def __init__(self, base_url: str = "", headers: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers
        self.client = httpx.Client(base_url=base_url, headers=headers, timeout=60.0)
        self.aclient = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0)

    def _handle_error(self, e: Exception, context: str = ""):
        """Map httpx errors to AIClient exceptions."""
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            try:
                # If streaming, we might need to read the response explicitly 
                # to get the error text, as it hasn't been consumed yet
                if not hasattr(e.response, "_content") and not e.response.is_closed:
                    content = e.response.read()
                else:
                    # In sync stream, it might be tricky if context manager active...
                    # but e.response.text uses .content
                    pass
            except Exception:
                pass
                
            # Safely access text (httpx will error if not read and not streaming closed properly)
            try:
                error_body = e.response.text
            except httpx.ResponseNotRead:
                # Force read if we catch it here? Or just fallback.
                # Usually if raise_for_status() triggers, we are in a state where we can read it.
                # BUT for stream(), we are inside the 'with' block.
                try:
                    e.response.read()
                    error_body = e.response.text
                except Exception:
                    error_body = "<Could not read error body>"
            
            error_msg = f"{context}: {error_body}"
            logger.error(f"HTTP Error {status}: {error_msg}")
            
            if status == 401 or status == 403:
                raise AuthenticationError(error_msg) from e
            elif status == 429:
                raise RateLimitError(error_msg) from e
            elif status == 400:
                raise InvalidRequestError(error_msg) from e
            elif status >= 500:
                raise ProviderError(error_msg) from e
            else:
                raise AIClientError(f"HTTP {status}: {error_msg}") from e
                
        elif isinstance(e, httpx.RequestError):
            logger.error(f"Network Error: {e}")
            raise NetworkError(f"Network error: {e}") from e
        else:
            logger.error(f"Unexpected Error: {e}")
            raise AIClientError(f"Unexpected error: {e}") from e

    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"SEND {endpoint} payload={data}")
        try:
            response = self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e, "Sync send failed")

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"ASYNC SEND {endpoint} payload={data}")
        try:
            response = await self.aclient.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e, "Async send failed")

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        logger.debug(f"STREAM {endpoint} payload={data}")
        try:
            with self.client.stream("POST", endpoint, json=data) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield {"raw": line}
        except Exception as e:
            self._handle_error(e, "Stream failed")

    async def stream_async(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        logger.debug(f"ASYNC STREAM {endpoint} payload={data}")
        try:
            async with self.aclient.stream("POST", endpoint, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield {"raw": line}
        except Exception as e:
            self._handle_error(e, "Async stream failed")
