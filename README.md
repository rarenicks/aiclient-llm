# aiclient

A minimal, production-minded Python SDK for interacting with AI models, streams, tools, and agentic systems.

## Key Features
- **Multi-Vendor**: unified interface for OpenAI, Anthropic, Google Gemini, and xAI.
- **Minimalist**: No heavy framework bloat. Just clean abstractions over HTTP.
- **Typed**: Pydantic models for responses and tools.
- **Zero-Config**: usage with `.env`.

## Installation

```bash
pip install -e .
```

## Configuration

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
XAI_API_KEY=...
```

## Quick Start

```python
from aiclient import Client

# Automatically loads keys from .env
client = Client()

# OpenAI
print(client.chat("gpt-4").generate("Hello OpenAI").text)

# Anthropic
print(client.chat("claude-3-opus").generate("Hello Claude").text)

# Google
print(client.chat("gemini-2.0-flash-exp").generate("Hello Gemini").text)

# xAI
print(client.chat("grok-2-latest").generate("Hello Grok").text)
```

## Usage Statistics
Access standardized usage stats across providers:

```python
response = client.chat("gpt-4").generate("Count to 5")
print(response.usage)
# Output: input_tokens=10 output_tokens=5 total_tokens=15
```

## Tools & Agents

```python
from aiclient.tools.base import Tool
from pydantic import BaseModel

class EchoSchema(BaseModel):
    message: str

def echo(message: str):
    return message

tool = Tool("echo", echo, EchoSchema)
print(tool.run(message="test"))
```
