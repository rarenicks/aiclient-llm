import pytest
from unittest.mock import MagicMock, patch
from aiclient.models.chat import ChatModel
from aiclient.transport.base import Transport
from aiclient.data_types import UserMessage, ModelResponse, Usage

class MockException(Exception):
    def __init__(self, status_code):
        self.response = MagicMock()
        self.response.status_code = status_code

class MockTransport(Transport):
    def __init__(self):
        self.call_count = 0
        self.fail_count = 0
    
    def send(self, endpoint, data):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise MockException(429)
        return {"choices": [{"message": {"content": "Success"}}]}

    async def send_async(self, endpoint, data):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise MockException(500)
        return {"choices": [{"message": {"content": "Success"}}]}
        
    def stream(self, endpoint, data):
        yield {"raw": "data"}

    async def stream_async(self, endpoint, data):
        yield {"raw": "data"}

def test_retry_sync():
    """Test standard 429 retry logic."""
    transport = MockTransport()
    transport.fail_count = 2 # Fail twice, succeed third time
    
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(text="Success", raw={})
    
    # Init model with retries=2
    model = ChatModel("gpt-test", provider, transport, max_retries=3, retry_delay=0.01)
    
    resp = model.generate("hello")
    print(f"Call count: {transport.call_count}")
    
    assert transport.call_count == 3
    assert resp.text == "Success"

import asyncio

def test_retry_async_wrapper():
    """Wrapper to run async test."""
    asyncio.run(_test_retry_async())

async def _test_retry_async():
    """Test async 500 retry logic."""
    transport = MockTransport()
    transport.fail_count = 2 
    
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(text="Success", raw={})
    
    model = ChatModel("gpt-test", provider, transport, max_retries=3, retry_delay=0.01)
    
    resp = await model.generate_async("hello")
    
    assert transport.call_count == 3
    assert resp.text == "Success"

def test_retry_fail_max():
    """Test failing after max retries."""
    transport = MockTransport()
    transport.fail_count = 5 # More than max_retries
    
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    
    model = ChatModel("gpt-test", provider, transport, max_retries=2, retry_delay=0.01)
    
    with pytest.raises(MockException):
        model.generate("hello")
        
    assert transport.call_count == 3 # Initial + 2 retries
