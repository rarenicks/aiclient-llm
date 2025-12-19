from typing import Any, Dict, Protocol, Tuple, Optional, Union, List
from ..types import ModelResponse, StreamChunk, BaseMessage, UserMessage

class Provider(Protocol):
    """
    Protocol that defines how to interact with an AI provider.
    It handles standardizing the request payload and parsing the response.
    """
    
    @property
    def base_url(self) -> str:
        """Return the base URL for this provider."""
        ...
        
    @property
    def headers(self) -> Dict[str, str]:
        """Return the headers for this provider."""
        ...

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the endpoint and JSON payload for the request.
        Returns (endpoint, json_payload).
        """
        ...
    
    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        """Parse the raw response into a standardized ModelResponse."""
        ...
        
    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        """Parse a stream chunk into a standardized StreamChunk."""
        ...
