
import pytest
import httpx
from unittest.mock import MagicMock, patch
from aiclient.client import Client
from aiclient.transport.http import HTTPTransport
from aiclient.providers.openai import OpenAIProvider

def test_default_client_timeout():
    """Test that client initialization sets the default timeout on transport."""
    client = Client(openai_api_key="mock", timeout=45.0)
    provider, _ = client._get_provider("gpt-4o")
    
    # Manually create transport to verify init params
    transport = client.transport_factory(
        base_url=provider.base_url, 
        headers=provider.headers, 
        timeout=client.timeout
    )
    
    assert transport.timeout == 45.0
    assert transport.client.timeout.read == 45.0
    assert transport.aclient.timeout.read == 45.0

def test_override_timeout_generate():
    """Test that passing timeout to generate() overrides the default."""
    client = Client(openai_api_key="mock", timeout=10.0)
    
    # Mock the internal transport.send method to check arguments
    with patch("aiclient.transport.http.HTTPTransport.send") as mock_send:
        # We need to mock response or send will crash trying to return
        mock_send.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {}
        }
        
        client.chat("gpt-4o").generate("Hello", timeout=5.0)
        
        # Verify call args
        args, kwargs = mock_send.call_args
        assert kwargs["timeout"] == 5.0

@pytest.mark.asyncio
async def test_override_timeout_generate_async():
    """Test specific timeout override on async generation."""
    client = Client(openai_api_key="mock", timeout=10.0)
    
    with patch("aiclient.transport.http.HTTPTransport.send_async") as mock_send:
        mock_send.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {}
        }
        
        await client.chat("gpt-4o").generate_async("Hello", timeout=2.5)
        
        args, kwargs = mock_send.call_args
        assert kwargs["timeout"] == 2.5

def test_no_timeout_override_uses_default():
    """Test that not passing timeout uses the client default (implicitly via httpx client)."""
    client = Client(openai_api_key="mock", timeout=20.0)
    
    with patch("aiclient.transport.http.HTTPTransport.send") as mock_send:
        mock_send.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {}
        }
        
        client.chat("gpt-4o").generate("Hello")
        
        # kwargs should NOT contain timeout if we didn't pass it, 
        # allowing the underlying httpx client (configured with default) to handle it.
        # OR our implementation passes None/nothing.
        # Our implementation passes `timeout=None` if not provided? 
        # No, update: if timeout is None, we don't put it in kwargs.
        
        args, kwargs = mock_send.call_args
        assert "timeout" not in kwargs or kwargs["timeout"] is None

