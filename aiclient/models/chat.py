from typing import Iterator, Dict, Any
from ..types import ModelResponse
from ..transport.base import Transport
from ..providers.base import Provider

class ChatModel:
    """Wrapper for chat model interactions using a Provider strategy."""
    def __init__(self, model_name: str, provider: Provider, transport: Transport):
        self.model_name = model_name
        self.provider = provider
        self.transport = transport

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response synchronously."""
        endpoint, data = self.provider.prepare_request(self.model_name, prompt)
        response_data = self.transport.send(endpoint, data)
        return self.provider.parse_response(response_data)

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream a response synchronously."""
        endpoint, data = self.provider.prepare_request(self.model_name, prompt)
        for chunk_data in self.transport.stream(endpoint, data):
            chunk = self.provider.parse_stream_chunk(chunk_data)
            if chunk:
                yield chunk.text

class SimpleResponse:
    def __init__(self, text: str, raw: Dict[str, Any]):
        self.text = text
        self.raw = raw
