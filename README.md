# aiclient-llm

[![PyPI version](https://img.shields.io/pypi/v/aiclient-llm.svg)](https://pypi.org/project/aiclient-llm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/aiclient-llm.svg)](https://pypi.org/project/aiclient-llm/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/aiclient-llm)](https://pepy.tech/project/aiclient-llm)

**A minimal, unified, and resilient Python client for modern LLMs.**

Supports **OpenAI**, **Anthropic** (Claude 3), **Google** (Gemini), and **xAI** (Grok) with a single, consistent interface.



## Documentation

- [**Getting Started**](docs/getting_started.md): Installation, Configuration, Basic Usage.
- [**Features Guide**](docs/features.md): Agents, Multimodal, Local LLMs (Ollama), Structured Output.
- [**Middleware**](docs/middleware.md): Cost tracking, logging, resilience, and custom middleware.
- [**Memory**](docs/memory.md): Conversation history management and persistence.
- [**Testing**](docs/testing.md): Mock providers and testing utilities.
- [**Error Handling**](docs/errors.md): Exception types and debugging.
- [**Examples**](examples/): Runnable demo scripts for new features.

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

## Installation

```bash
pip install aiclient-llm
```

## Quick Start

### Basic Chat

```python
from aiclient import Client

client = Client(
    api_key_openai="sk-...", 
    api_key_anthropic="sk-ant-..."
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
from aiclient.types import UserMessage, Text, Image

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

### Retries

```python
# Retries up to 3 times with backoff
client = Client(max_retries=3, retry_delay=1.0)
```

## License

Apache-2.0
