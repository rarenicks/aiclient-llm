"""
Tests for aiclient.testing module.
"""
import pytest
from aiclient.testing import MockProvider
from aiclient import Client
from aiclient.types import UserMessage

def test_mock_provider_responses():
    """Test MockProvider queues and returns responses."""
    provider = MockProvider()
    provider.add_response("Response 1")
    provider.add_response("Response 2")
    
    # Check queue
    assert len(provider._responses) == 2
    
    # Parse 1
    resp1 = provider.parse_response({})
    assert resp1.text == "Response 1"
    
    # Parse 2
    resp2 = provider.parse_response({})
    assert resp2.text == "Response 2"
    
    # Empty queue default
    resp3 = provider.parse_response({})
    assert resp3.text == "Mock Response"

def test_mock_provider_requests_capture():
    """Test MockProvider captures request details."""
    provider = MockProvider()
    messages = [UserMessage(content="Hello")]
    
    endpoint, data = provider.prepare_request(
        "gpt-4o", 
        messages, 
        stream=True
    )
    
    assert endpoint == "mock://test"
    assert len(provider.requests) == 1
    req = provider.requests[0]
    assert req["model"] == "gpt-4o"
    assert req["stream"] is True
    # Messages are serialized
    assert req["messages"][0]["content"] == "Hello"
    assert req["messages"][0]["role"] == "user"

def test_mock_provider_errors():
    """Test MockProvider can raise errors."""
    provider = MockProvider()
    provider.add_error(ValueError("Mock Error"))
    
    with pytest.raises(ValueError, match="Mock Error"):
        provider.parse_response({})
