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

## Structured Output
Generate validated JSON objects directly using Pydantic models. Works across all providers.

```python
from pydantic import BaseModel
from aiclient import Client

class User(BaseModel):
    name: str
    age: int

client = Client()
user = client.chat("gpt-4").generate("Who is Alice?", response_model=User)

print(f"Name: {user.name}, Age: {user.age}")
# Output: Name: Alice, Age: 30
```

## Conversation History
Manage multi-turn conversations using typed Message objects.

```python
from aiclient.types import SystemMessage, UserMessage, AssistantMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="Hello"),
    AssistantMessage(content="Hi there!"),
    UserMessage(content="What is your name?")
]

response = client.chat("claude-3-opus").generate(messages)
print(response.text)
```

## Middleware & Observability
Inject custom logic before requests or after responses. Useful for logging, cost tracking, or prompt engineering.

### Cost Tracking Example
```python
from aiclient.middleware import CostTrackingMiddleware

tracker = CostTrackingMiddleware()

client = Client()
client.add_middleware(tracker)

client.chat("gpt-4").generate("Hello")
print(f"Total Tokens: {tracker.total_input_tokens} in, {tracker.total_output_tokens} out")
```

## Async & Streaming
Complete async support for high-throughput applications.

### Async Generation
```python
import asyncio
from aiclient import Client

async def main():
    client = Client()
    response = await client.chat("gpt-4").generate_async("Hello Async World")
    print(response.text)

asyncio.run(main())
```

### Streaming (Sync & Async)
Stream responses token-by-token.

```python
# Synchronous Streaming
for chunk in client.chat("gpt-4").stream("Count to 5"):
    print(chunk, end="", flush=True)

# Asynchronous Streaming
async for chunk in client.chat("gpt-4").stream_async("Count to 5"):
    print(chunk, end="", flush=True)
```

## Tools & Function Calling
Seamlessly execute client-side tools driven by the LLM.

```python
from aiclient.tools import Tool
from pydantic import BaseModel

# 1. Define Tool Schema & Function
class WeatherInput(BaseModel):
    city: str

def get_weather(city: str):
    return f"Sunny in {city}"

# 2. Create Tool
weather_tool = Tool(name="get_weather", fn=get_weather, schema=WeatherInput)

# 3. Pass to LLM
response = client.chat("gpt-4").generate(
    "What's the weather in Paris?",
    tools=[weather_tool]
)

# 4. Handle Tool Call
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call.name}, Args: {call.arguments}")
        # Execute tool
        if call.name == "get_weather":
            result = weather_tool.run(**call.arguments)
            print(f"Result: {result}")
```
