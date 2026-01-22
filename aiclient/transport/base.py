from typing import Any, AsyncIterator, Dict, Iterator, Protocol


class Transport(Protocol):
    """
    Abstract interface for network transport.
    """

    def send(self, endpoint: str, data: Dict[str, Any], timeout: float = None) -> Dict[str, Any]:
        """Send a synchronous request."""
        ...

    async def send_async(self, endpoint: str, data: Dict[str, Any], timeout: float = None) -> Dict[str, Any]:
        """Async version of send."""
        ...

    def stream(self, endpoint: str, data: Dict[str, Any], timeout: float = None) -> Iterator[Dict[str, Any]]:
        """Stream a synchronous request."""
        ...

    async def stream_async(
        self, endpoint: str, data: Dict[str, Any], timeout: float = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream an asynchronous request."""
        ...
