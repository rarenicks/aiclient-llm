# aiclient-llm

![AIClient Banner](https://raw.githubusercontent.com/rarenicks/aiclient/main/assets/aiclient_banner.jpeg)

[![PyPI version](https://img.shields.io/pypi/v/aiclient-llm.svg)](https://pypi.org/project/aiclient-llm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/aiclient-llm.svg)](https://pypi.org/project/aiclient-llm/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/aiclient-llm)](https://pepy.tech/project/aiclient-llm)

**A minimal, unified, and resilient Python client for modern LLMs.**

Supports **OpenAI**, **Anthropic** (Claude 3), **Google** (Gemini), and **xAI** (Grok) with a single, consistent interface.



## Documentation

- [**Getting Started**](https://github.com/rarenicks/aiclient/blob/main/docs/getting_started.md): Installation, Configuration, Basic Usage.
- [**Features Guide**](https://github.com/rarenicks/aiclient/blob/main/docs/features.md): Agents, Multimodal, Local LLMs (Ollama), Structured Output.
- [**Middleware**](https://github.com/rarenicks/aiclient/blob/main/docs/middleware.md): Cost tracking, logging, resilience, and custom middleware.
- [**Memory**](https://github.com/rarenicks/aiclient/blob/main/docs/memory.md): Conversation history management and persistence.
- [**Testing**](https://github.com/rarenicks/aiclient/blob/main/docs/testing.md): Mock providers and testing utilities.
- [**Error Handling**](https://github.com/rarenicks/aiclient/blob/main/docs/errors.md): Exception types and debugging.
- [**Examples**](https://github.com/rarenicks/aiclient/blob/main/examples/): Runnable demo scripts for new features.

## Key Features

- 🦄 **Unified API**: Works with OpenAI, Anthropic, Google Gemini, and Ollama.
- ⚡ **Streaming Support**: Real-time responses with a simple iterator interface.
- 👁️ **Multimodal (Vision)**: Send images (paths, URLs, base64) to vision-capable models.
- 🚀 **Prompt Caching**: Native support for Anthropic Prompt Caching headers.
- 🏗️ **Structured Outputs**: Native strict JSON Schema support for OpenAI.
- 🛡️ **Resilient**: Circuit Breakers, Rate Limiters, and automatic retries.
- 🔭 **Observability**: Tracing and OpenTelemetry hooks.
- 🤖 **Agent Primitives**: Built-in ReAct loop for tool-using agents.
- 🔌 **Model Context Protocol (MCP)**: Connect to 16K+ external tools (GitHub, Postgres, filesystem).
- 📊 **Middleware**: Inspect requests, track costs, or log data.
- 🧠 **Memory Management**: Built-in conversation history with token-aware truncation
- 🧪 **Testing Utilities**: Mock providers for deterministic unit tests
- 📦 **Batch Processing**: Efficiently process thousands of requests concurrently
- 🛡️ **Type-Safe Errors**: Specific exception types for better error handling

## Architecture at a Glance

![aiclient-llm Architecture](https://raw.githubusercontent.com/rarenicks/aiclient/main/assets/architecture.png)

## Installation

```bash
pip install aiclient-llm
```

## Quick Start

### Basic Chat

```python
from aiclient import Client

client = Client(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-..."
)

# Call OpenAI
response = client.chat("gpt-4o").generate("Hello!")
print(response.text)

# Call Claude
response = client.chat("claude-3-opus-20240229").generate("Hello!")
print(response.text)
```

### Multimodal (Vision)

```python
from aiclient.data_types import UserMessage, Text, Image

msg = UserMessage(content=[
    Text(text="What's in this image?"),
    Image(path="./image.png") # Handles base64 automatically
])

response = client.chat("gpt-4o").generate([msg])
print(response.text)
```

### Agents (Tool Use)

```python
from aiclient.agent import Agent

def get_weather(location: str):
    return "Sunny in " + location

agent = Agent(
    model=client.chat("gpt-4o"),
    tools=[get_weather]
)

print(agent.run("Weather in SF?"))
```

### MCP Integration 🔌

Connect to external tools using the Model Context Protocol.

```python
agent = Agent(
    model=client.chat("gpt-4o"),
    mcp_servers={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
        }
    }
)

# Agent can now use file system tools!
print(agent.run("List all Python files in the current directory"))
```

### Local LLMs (Ollama) 🏠

Use the `provider:model` syntax to route requests to local models (e.g., via Ollama).

```python
# Connects to http://localhost:11434/v1 by default
client.chat("ollama:llama3").generate("Why is the sky blue?")

# Connect to custom URL (e.g. LMStudio)
client = Client(ollama_base_url="http://localhost:1234/v1")
client.chat("ollama:mistral").generate("Hi")
```

### Streaming

```python
for chunk in client.chat("gpt-4o").stream("Write a poem"):
    print(chunk.text, end="", flush=True)
```

## Configuration

### Embeddings

```python
# Generate embeddings using the unified interface
vector = await client.embed("Hello world", model="text-embedding-3-small")

# Batch generation
vectors = await client.embed_batch(["Hello", "World"], model="text-embedding-3-small")
```

### Structured Outputs

```python
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    class_type: str

# Guaranteed JSON response
char = client.chat("gpt-4o").generate(
    "Create a wizard",
    response_model=Character
)
print(char.name)
```

## Production Resilience 🛡️

### Circuit Breakers
Prevent cascade failures when a provider is down.

```python
from aiclient import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
client.add_middleware(cb)
```

### Rate Limiters
Respect API rate limits automatically.

```python
from aiclient import RateLimiter

rl = RateLimiter(requests_per_minute=60)
client.add_middleware(rl)
```

### Fallback Chains
Automatically ensure high availability.

```python
from aiclient import FallbackChain

fallback = FallbackChain(client, ["gpt-4o", "claude-3-opus", "gemini-1.5-pro"])
response = fallback.generate("Critical query")
```

## Observability 🔭

### Cost Tracking
Track spending in real-time across all providers.

```python
from aiclient import CostTrackingMiddleware

tracker = CostTrackingMiddleware()
client.add_middleware(tracker)

# ... after requests ...
print(f"Total Cost: ${tracker.total_cost_usd:.4f}")
```

### Logging & OpenTelemetry
Full visibility into your AI calls.

```python
from aiclient import LoggingMiddleware, OpenTelemetryMiddleware

# Redact API keys from logs automatically
client.add_middleware(LoggingMiddleware(redact_keys=True))

# Export traces to Jaeger/Zipkin/etc
client.add_middleware(OpenTelemetryMiddleware(service_name="my-app"))
```

## Advanced Features

### Semantic Caching
Save money by caching responses based on meaning.

```python
from aiclient import SemanticCacheMiddleware

cache = SemanticCacheMiddleware(embedder=my_embedder, threshold=0.9)
client.add_middleware(cache)
```

### Batch Processing
Efficiently process thousands of requests.

```python
results = await client.batch(
    ["Q1", "Q2", "Q3"],
    process_func,
    concurrency=10
)
```

## Testing 🧪

Write deterministic unit tests without API keys.

```python
from aiclient import MockProvider

def test_feature():
    provider = MockProvider()
    provider.add_response("Mocked AI response")
    
    # Client will use this response instead of hitting API
    response = provider.parse_response({})
    assert response.text == "Mocked AI response"
```

## License

Apache-2.0
