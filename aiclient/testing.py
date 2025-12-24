"""
Testing utilities for aiclient applications.
Use these tools to verify your code without making real API calls.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib
from .providers.base import Provider
from .transport.base import Transport
from .types import ModelResponse, BaseMessage, StreamChunk, Usage

class MockTransport(Transport):
    """
    A transport that does nothing but return empty data.
    Used in conjunction with MockProvider which handles the response queue.
    """
    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def stream(self, endpoint: str, data: Dict[str, Any]):
        yield {}

    async def stream_async(self, endpoint: str, data: Dict[str, Any]):
        yield {}


class MockProvider(Provider):
    """
    A provider that returns pre-configured responses.
    Useful for unit testing your application logic.
    """
    def __init__(self, base_url: str = "mock://test"):
        self._base_url = base_url
        self._responses: List[Union[ModelResponse, Exception]] = []
        self._stream_chunks: List[List[StreamChunk]] = []
        self.requests: List[Dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {"X-Mock": "true"}

    def add_response(self, text: str, raw: Dict[str, Any] = None):
        """Add a canned response to the queue."""
        self._responses.append(ModelResponse(
            text=text,
            raw=raw or {},
            usage=Usage(total_tokens=10),
            provider="mock"
        ))

    def add_error(self, error: Exception):
        """Add an error to be raised during generation."""
        self._responses.append(error)

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False, response_schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Log the request and return dummy data."""
        # Serialize messages to dicts for JSON safety
        serialized_messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
        
        request_data = {
            "model": model,
            "messages": serialized_messages,
            "tools": tools,
            "stream": stream,
            "response_schema": response_schema,
            "strict": strict
        }
        self.requests.append(request_data)
        return self.base_url, request_data

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        """Return the next queued response."""
        if not self._responses:
            # Default response if none queued
            return ModelResponse(
                text="Mock Response",
                raw={},
                usage=Usage(),
                provider="mock"
            )
        
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        """Parse stream chunk (not fully implemented for generic mock yet)."""
        return StreamChunk(text=chunk.get("text", ""), delta=chunk.get("text", ""))

@contextlib.contextmanager
def capture_on_error():
    """Context manager to capture errors (placeholder for future middleware testing)."""
    errors = []
    yield errors
