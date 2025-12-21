# aiclient-llm

[![PyPI version](https://badge.fury.io/py/aiclient-llm.svg)](https://badge.fury.io/py/aiclient-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A minimal, unified, and resilient Python client for modern LLMs.**

Supports **OpenAI**, **Anthropic** (Claude 3), **Google** (Gemini), and **xAI** (Grok) with a single, consistent interface.

## Features

- 🦄 **Unified Interface**: Swap between providers without changing code.
- ⚡ **Async & Sync**: Native asyncio support for high-performance apps.
- 👁️ **Multimodal**: Send images and text seamlessly.
- 🛡️ **Resilient**: Automatic retries with exponential backoff for 429/5xx errors.
- 🤖 **Agent Primitives**: Built-in ReAct loop for tool-using agents.
- 🛠️ **Tool Calling**: Standardized function calling across providers.
- 📊 **Middleware**: Inspect requests, track costs, or log data.

## Installation

```bash
pip install aiclient-llm
```

*(Note: Not yet on PyPI, install from source/git)*

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

MIT
