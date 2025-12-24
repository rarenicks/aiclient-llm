# Features

## Local LLMs (Ollama & LMStudio) 🏠

`aiclient-llm` supports routing requests to local inference servers using the `provider:model` syntax.

### Usage
Default (Ollama):
```python
# Connects to http://localhost:11434/v1
response = client.chat("ollama:llama3").generate("Hi")
```

Custom URL (e.g., LMStudio, remote server):
```python
client = Client(ollama_base_url="http://192.168.1.5:1234/v1")
response = client.chat("ollama:mistral").generate("Hi")
```

## Agents & Tool Use 🤖

Build powerful agents using the `Agent` class, which implements a ReAct loop.

```python
from aiclient import Client, Agent

def get_weather(city: str):
    return f"Sunny in {city}"

client = Client()
agent = Agent(
    model=client.chat("gpt-4o"),
    tools=[get_weather],
    max_steps=10
)

result = agent.run("What's the weather in Tokyo?")
print(result)
```

## Multimodal (Vision) 👁️

Send images easily to models that support it (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5).

`aiclient` handles image formatting automatically for each provider. You can provide images via:
- **Local Path**: `Image(path="./image.png")`
- **URL**: `Image(url="https://...")`
- **Base64**: `Image(base64_data="...")`

```python
from aiclient.types import UserMessage, Text, Image

# Mixed content: Text + Images
msg = UserMessage(content=[
    Text(text="Analyze this diagram and compare it with the reference."),
    Image(path="./diagram.png"),                # Local file
    Image(url="https://example.com/ref.jpg")    # Remote URL
])

response = client.chat("gpt-4o").generate([msg])
print(response.text)
```

## Streaming Responses ⚡
 
Receive tokens as they are generated for a responsive UI experience.
 
```python
# Sync Streaming
for chunk in client.chat("gpt-4o").stream("Write a long story"):
    print(chunk, end="", flush=True)
 
# Async Streaming
async for chunk in client.chat("gpt-4o").stream_async("Write a long story"):
    print(chunk, end="", flush=True)
```
 
## Prompt Caching (Cost Optimization) 💰

Reduce costs by up to 90% and latency by up to 85% with prompt caching. Currently supported on Anthropic (Claude 3.5 Sonnet/Haiku/Opus).

### Usage

Mark cache breakpoints using the `cache_control` parameter on messages.

```python
from aiclient import Client, SystemMessage, UserMessage

client = Client(api_key_anthropic="...")

messages = [
    # Cache system prompt (e.g. large context docs)
    SystemMessage(
        content="<long text field>...", 
        cache_control="ephemeral"
    ),
    # Cache up to the last turn of conversation
    UserMessage(content="Hello", cache_control="ephemeral")
]

response = client.chat("claude-3-5-sonnet-20240620").generate(messages)

# Check savings
print(f"Cache Creation Tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache Read Tokens: {response.usage.cache_read_input_tokens}")
```

## Structured Output (Pydantic) 📦

Validate responses against a schema. v0.2.0 adds support for **Native Structured Outputs** (e.g., OpenAI `response_format`), guaranteeing 100% schema adherence.

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# 1. Native Restricted Mode (strict=True) - OpenAI only for now
user = client.chat("gpt-4o").generate(
    "John is 25 years old", 
    response_model=UserInfo,
    strict=True
)

# 2. Universal Fallback (strict=False) - Works on all providers via prompt injection
user = client.chat("claude-3-opus").generate(
    "John is 25 years old", 
    response_model=UserInfo
)

print(user.name, user.age)
```

## Batch Processing 📦

Process thousands of requests concurrently with automatic rate limiting and error handling.

### Usage

```python
from aiclient import Client

client = Client()

# Define inputs
prompts = [
    "Translate 'hello' to Spanish",
    "Translate 'goodbye' to French",
    "Translate 'thank you' to German"
]

# Process concurrently
async def translate(prompt: str):
    return await client.chat("gpt-4o-mini").generate_async(prompt)

# Run batch with 5 concurrent requests
results = await client.batch(
    inputs=prompts,
    func=translate,
    concurrency=5,           # Max 5 requests at once
    return_exceptions=True   # Return errors instead of raising
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Request {i} failed: {result}")
    else:
        print(f"Result {i}: {result.text}")
```

### Use Cases

- **Data Labeling**: Classify or label thousands of records
- **Content Generation**: Generate descriptions, summaries, or translations in bulk
- **Evaluation**: Test prompts across multiple inputs
- **ETL Pipelines**: Process large datasets with LLM augmentation

## Model Context Protocol (MCP) 🔌

Connect to external tools (GitHub, Postgres, etc.) using the standard Model Context Protocol.

### Usage

```python
from aiclient import Client, Agent

client = Client()

# Define MCP servers
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
    }
}

# Agent with MCP tools
agent = Agent(
    model=client.chat("gpt-4o"),
    mcp_servers=mcp_config
)

# Agent can now use file system tools!
result = agent.run("List files in the current directory")
print(result)
```
