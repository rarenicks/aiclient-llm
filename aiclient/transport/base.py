from typing import Any, Dict, Protocol, Optional, AsyncIterator, Iterator
from ..types import ModelResponse, StreamChunk

class Transport(Protocol):
    """
    Abstract interface for network transport.
    """
    
    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a synchronous request."""
        ...

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send an asynchronous request."""
        ...

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream a synchronous request."""
        ...

    async def stream_async(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Stream an asynchronous request."""
        ...
