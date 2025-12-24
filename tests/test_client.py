"""
Tests for Client provider routing and configuration.
"""
import pytest
from aiclient import Client
from aiclient.providers.openai import OpenAIProvider
from aiclient.providers.anthropic import AnthropicProvider
from aiclient.providers.google import GoogleProvider
from aiclient.providers.ollama import OllamaProvider
from aiclient.middleware import CostTrackingMiddleware


def test_client_resolves_openai_by_prefix():
    """Test client routes gpt-* models to OpenAI."""
    client = Client(openai_api_key="sk-test")
    model = client.chat("gpt-4o")

    assert isinstance(model.provider, OpenAIProvider)
    assert model.model_name == "gpt-4o"


def test_client_resolves_anthropic_by_prefix():
    """Test client routes claude-* models to Anthropic."""
    client = Client(anthropic_api_key="sk-ant-test")
    model = client.chat("claude-3-opus")

    assert isinstance(model.provider, AnthropicProvider)
    assert model.model_name == "claude-3-opus"


def test_client_resolves_google_by_prefix():
    """Test client routes gemini-* models to Google."""
    client = Client(google_api_key="test-key")
    model = client.chat("gemini-1.5-pro")

    assert isinstance(model.provider, GoogleProvider)
    assert model.model_name == "gemini-1.5-pro"


def test_client_resolves_xai_by_prefix():
    """Test client routes grok-* models to xAI."""
    client = Client(xai_api_key="xai-test")
    model = client.chat("grok-beta")

    assert isinstance(model.provider, OpenAIProvider)
    assert model.provider.base_url == "https://api.x.ai/v1"
    assert model.model_name == "grok-beta"


def test_client_explicit_provider_syntax():
    """Test client supports provider:model syntax."""
    client = Client(openai_api_key="sk-test")

    # Explicit OpenAI
    model = client.chat("openai:gpt-4")
    assert isinstance(model.provider, OpenAIProvider)
    assert model.model_name == "gpt-4"

    # Explicit Anthropic
    client = Client(anthropic_api_key="sk-ant-test")
    model = client.chat("anthropic:claude-3-opus")
    assert isinstance(model.provider, AnthropicProvider)
    assert model.model_name == "claude-3-opus"


def test_client_ollama_provider():
    """Test client routes ollama:* to Ollama provider."""
    client = Client()
    model = client.chat("ollama:llama3")

    assert isinstance(model.provider, OllamaProvider)
    assert model.model_name == "llama3"
    assert model.provider.base_url == "http://localhost:11434/v1"


def test_client_ollama_custom_url():
    """Test client supports custom Ollama URL."""
    client = Client(ollama_base_url="http://192.168.1.100:11434/v1")
    model = client.chat("ollama:mistral")

    assert isinstance(model.provider, OllamaProvider)
    assert model.provider.base_url == "http://192.168.1.100:11434/v1"


def test_client_unknown_model_raises():
    """Test client raises error for unknown models."""
    client = Client()

    with pytest.raises(ValueError) as exc:
        client.chat("unknown-model-xyz")

    assert "Unknown model provider" in str(exc.value)
    assert "provider:model_name" in str(exc.value)


def test_client_add_middleware():
    """Test client can add middleware that propagates to models."""
    client = Client(openai_api_key="sk-test")
    tracker = CostTrackingMiddleware()

    client.add_middleware(tracker)

    model = client.chat("gpt-4o")

    # Middleware should be in model's middleware list
    assert tracker in model.middlewares


def test_client_retry_configuration():
    """Test client retry configuration propagates to models."""
    client = Client(
        openai_api_key="sk-test",
        max_retries=5,
        retry_delay=2.0
    )

    model = client.chat("gpt-4o")

    assert model.max_retries == 5
    assert model.retry_delay == 2.0


def test_client_env_var_loading():
    """Test client loads API keys from environment variables."""
    import os

    # Temporarily set env vars
    os.environ["OPENAI_API_KEY"] = "env-key-123"

    client = Client()  # No explicit keys

    model = client.chat("gpt-4o")
    assert model.provider.api_key == "env-key-123"

    # Clean up
    del os.environ["OPENAI_API_KEY"]


def test_client_explicit_key_overrides_env():
    """Test explicit API keys override environment variables."""
    import os

    os.environ["OPENAI_API_KEY"] = "env-key"

    client = Client(openai_api_key="explicit-key")

    model = client.chat("gpt-4o")
    assert model.provider.api_key == "explicit-key"

    del os.environ["OPENAI_API_KEY"]


def test_client_multiple_models():
    """Test client can create multiple models."""
    client = Client(
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test"
    )

    model1 = client.chat("gpt-4o")
    model2 = client.chat("claude-3-opus")

    assert isinstance(model1.provider, OpenAIProvider)

def test_client_mock_streaming():
    """Test client streaming interface with mock provider."""
    from unittest.mock import MagicMock
    from aiclient.data_types import StreamChunk
    
    # Setup mock
    client = Client(openai_api_key="sk-test")
    model = client.chat("gpt-4o")
    
    # Mock the transport stream method
    mock_transport = MagicMock()
    # stream returns an iterator of raw chunks wrapped in {"raw": ...} as expected by OpenAIProvider
    import json
    chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
    chunk2 = {"choices": [{"delta": {"content": " World"}}]}
    
    # Provider expects: {"raw": "data: <json_string>"}
    mock_transport.stream.return_value = [
        {"raw": f"data: {json.dumps(chunk1)}"},
        {"raw": f"data: {json.dumps(chunk2)}"}
    ]
    model.transport = mock_transport

    # Execute
    chunks = list(model.stream("prompt"))
    
    assert len(chunks) == 2
    assert chunks[0] == "Hello" # stream yield strings, not objects
    assert chunks[1] == " World"

