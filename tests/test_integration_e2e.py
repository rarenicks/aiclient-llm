"""
End-to-end integration tests with real API calls.
These tests require API keys in .env file.
Skip if keys are not available.
"""
import pytest
import os
from dotenv import load_dotenv
from aiclient import Client
from aiclient.types import SystemMessage, UserMessage
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Check which API keys are available
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_GOOGLE = bool(os.getenv("GEMINI_API_KEY"))
HAS_XAI = bool(os.getenv("XAI_API_KEY"))

@pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
def test_openai_basic_generation():
    """Test basic OpenAI generation with real API."""
    client = Client()
    response = client.chat("gpt-4o-mini").generate("Say 'test passed' and nothing else.")

    assert response.text
    assert len(response.text) > 0
    assert response.usage
    assert response.usage.total_tokens > 0
    assert response.provider == "openai"
    print(f"✅ OpenAI test passed: {response.text}")

@pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_basic_generation():
    """Test basic Anthropic generation with real API."""
    client = Client()
    response = client.chat("claude-3-haiku-20240307").generate("Say 'test passed' and nothing else.")

    assert response.text
    assert len(response.text) > 0
    assert response.usage
    assert response.usage.total_tokens > 0
    assert response.provider == "anthropic"
    print(f"✅ Anthropic test passed: {response.text}")

@pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_prompt_caching_e2e():
    """Test prompt caching with real Anthropic API."""
    client = Client()

    # Large system prompt to trigger caching
    # Large system prompt to trigger caching (needs >1024 tokens)
    system_text = "You are a helpful assistant. " * 500

    messages = [
        SystemMessage(content=system_text, cache_control="ephemeral"),
        UserMessage(content="Say 'hello'")
    ]

    # First call - should create cache
    resp1 = client.chat("claude-3-haiku-20240307").generate(messages)
    assert resp1.text
    print(f"✅ Cache creation tokens: {resp1.usage.cache_creation_input_tokens}")

    # Second call - should use cache
    resp2 = client.chat("claude-3-haiku-20240307").generate(messages)
    assert resp2.text
    print(f"✅ Cache read tokens: {resp2.usage.cache_read_input_tokens}")

    # At least one should have cache activity
    total_cache_tokens = (
        (resp1.usage.cache_creation_input_tokens or 0) +
        (resp2.usage.cache_read_input_tokens or 0)
    )
    assert total_cache_tokens > 0, "Expected cache activity"

@pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
def test_openai_structured_outputs_e2e():
    """Test structured outputs with real OpenAI API."""

    class SimpleResponse(BaseModel):
        color: str
        number: int

    client = Client()

    # Test with strict=True (native structured outputs)
    response = client.chat("gpt-4o-mini").generate(
        "Return color='blue' and number=42",
        response_model=SimpleResponse,
        strict=True
    )

    assert isinstance(response, SimpleResponse)
    assert response.color
    assert response.number
    print(f"✅ Structured output: color={response.color}, number={response.number}")

@pytest.mark.skipif(not HAS_GOOGLE, reason="GEMINI_API_KEY not set")
def test_google_basic_generation():
    """Test basic Google Gemini generation with real API."""
    client = Client()
    response = client.chat("gemini-1.5-flash").generate("Say 'test passed' and nothing else.")

    assert response.text
    assert len(response.text) > 0
    assert response.provider == "google"
    print(f"✅ Google test passed: {response.text}")

@pytest.mark.skipif(not HAS_XAI, reason="XAI_API_KEY not set")
def test_xai_basic_generation():
    """Test basic xAI Grok generation with real API."""
    client = Client()
    response = client.chat("grok-beta").generate("Say 'test passed' and nothing else.")

    assert response.text
    assert len(response.text) > 0
    print(f"✅ xAI test passed: {response.text}")

# Requires pytest-asyncio
# @pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
# @pytest.mark.asyncio
# async def test_openai_async_generation():
#     """Test async generation with real OpenAI API."""
#     client = Client()
#     model = client.chat("gpt-4o-mini")
#     response = await model.generate_async("Say 'async test passed'")
#
#     assert response.text
#     assert len(response.text) > 0
#     print(f"✅ Async test passed: {response.text}")

@pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
def test_openai_streaming():
    """Test streaming with real OpenAI API."""
    client = Client()
    model = client.chat("gpt-4o-mini")

    chunks = []
    for chunk in model.stream("Count to 3, just numbers separated by spaces"):
        chunks.append(chunk)
        print(f"Chunk: {chunk}", end="", flush=True)

    print()  # New line

    full_text = "".join(chunks)
    assert len(chunks) > 0
    assert len(full_text) > 0
    print(f"✅ Streaming test passed: {len(chunks)} chunks received")

# Requires pytest-asyncio
# @pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
# @pytest.mark.asyncio
# async def test_openai_async_streaming():
#     """Test async streaming with real OpenAI API."""
#     client = Client()
#     model = client.chat("gpt-4o-mini")
#
#     chunks = []
#     async for chunk in model.stream_async("Count to 3"):
#         chunks.append(chunk)
#
#     assert len(chunks) > 0
#     print(f"✅ Async streaming test passed: {len(chunks)} chunks received")
