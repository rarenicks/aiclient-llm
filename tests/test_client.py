import pytest
from typing import Any, Dict, Iterator, AsyncIterator
from aiclient.client import Client
from aiclient.transport.base import Transport
from aiclient.providers.openai import OpenAIProvider
from aiclient.providers.anthropic import AnthropicProvider

class MockTransport(Transport):
    def __init__(self, base_url: str = "", headers: Dict[str, str] = None, response_data: Dict[str, Any] = None):
        self.base_url = base_url
        self.headers = headers
        self.response_data = response_data or {}
        self.sent_data = []

    def send(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self.sent_data.append((self.base_url + endpoint, data, self.headers))
        return self.response_data

    async def send_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self.sent_data.append((self.base_url + endpoint, data, self.headers))
        return self.response_data

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield {"raw": "data: " + str(self.response_data)}

    async def stream_async(self, endpoint: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        yield {"raw": "data: " + str(self.response_data)}

def test_client_resolves_openai():
    client = Client(openai_api_key="sk-test", transport_factory=lambda **kwargs: MockTransport(**kwargs, response_data={
        "choices": [{"message": {"content": "Hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }))
    model = client.chat("gpt-4")
    assert isinstance(model.provider, OpenAIProvider)
    
    response = model.generate("Hi")
    assert response.text == "Hello"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 2
    assert response.usage.total_tokens == 7
    
    transport = model.transport
    assert transport.base_url == "https://api.openai.com/v1"
    assert transport.headers["Authorization"] == "Bearer sk-test"

def test_client_resolves_anthropic():
    client = Client(anthropic_api_key="sk-ant", transport_factory=lambda **kwargs: MockTransport(**kwargs, response_data={"content": [{"text": "Hello Claude"}]}))
    model = client.chat("claude-3")
    assert isinstance(model.provider, AnthropicProvider)
    
    response = model.generate("Hi")
    assert response.text == "Hello Claude"
    
    transport = model.transport
    assert transport.base_url == "https://api.anthropic.com/v1"
    assert "x-api-key" in transport.headers

def test_client_resolves_xai():
    client = Client(xai_api_key="xai-key", transport_factory=lambda **kwargs: MockTransport(**kwargs, response_data={"choices": [{"message": {"content": "I am Grok"}}] }))
    model = client.chat("grok-1")
    assert isinstance(model.provider, OpenAIProvider)
    assert model.provider.base_url == "https://api.x.ai/v1"
    
    response = model.generate("Hi")
    assert response.text == "I am Grok"
    
    transport = model.transport
    assert transport.base_url == "https://api.x.ai/v1"

