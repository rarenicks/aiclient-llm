# Middleware & Observability

Middleware allows you to inject custom logic before a request is sent to the LLM and after a response is received. This is useful for logging, cost tracking, prompt engineering, or PII masking.

## How it Works

A middleware is any class that implements the `Middleware` protocol:

```python
class Middleware(Protocol):
    def before_request(self, model: str, prompt: Union[str, List[BaseMessage]]) -> Union[str, List[BaseMessage]]:
        ...
    def after_response(self, response: ModelResponse) -> ModelResponse:
        ...
```

## Built-in Middleware

### CostTrackingMiddleware

Tracks estimated token usage and USD cost across all requests in a session.

```python
from aiclient import Client
from aiclient.middleware import CostTrackingMiddleware

# 1. Create tracker
tracker = CostTrackingMiddleware()

# 2. Attach to client
client = Client()
client.add_middleware(tracker)

# 3. Use client
client.chat("gpt-4o").generate("Hello")

# 4. Check stats
print(f"Total Cost: ${tracker.total_cost_usd:.4f}")
print(f"Tokens: {tracker.total_input_tokens} In / {tracker.total_output_tokens} Out")
```

## Creating Custom Middleware

### Example: Logging Middleware

```python
class LoggingMiddleware:
    def before_request(self, model, prompt):
        print(f"[LOG] Sending request to {model}")
        return prompt # Must return prompt
        
    def after_response(self, response):
        print(f"[LOG] Received {response.usage.total_tokens} tokens")
        return response # Must return response

client.add_middleware(LoggingMiddleware())
```

### Example: PII Redaction

```python
class PIIFilterMiddleware:
    def before_request(self, model, prompt):
        if isinstance(prompt, str):
            return prompt.replace("EMAIL_ADDRESS", "[REDACTED]")
        return prompt
        
    def after_response(self, response):
        return response
```
```

## Resilience Middleware 🛡️

Protect your application from downstream failures.
 
### RetryMiddleware
 
Automatically retry failed requests (e.g., 429 Rate Limit, 5xx Server Errors) with exponential backoff.
 
```python
from aiclient import RetryMiddleware
 
client.add_middleware(RetryMiddleware(
    max_retries=3,
    retry_delay=1.0,    # Initial delay in seconds
    backoff_factor=2.0, # Multiplier for each retry
    jitter=True         # Add random jitter to avoid thundering herd
))
```

### CircuitBreaker
Stops sending requests to a failing provider for a set time prevent cascading failures.

```python
from aiclient import CircuitBreaker

client.add_middleware(CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0  # seconds
))
```

### RateLimiter
Limits the number of requests per minute (RPM).

```python
from aiclient import RateLimiter

# Limit to 60 requests per minute
client.add_middleware(RateLimiter(requests_per_minute=60))
```

### FallbackChain

Automatically falls back to alternative models if the primary model fails. Useful for cost optimization, handling rate limits, or ensuring high availability.

```python
from aiclient import Client, FallbackChain

client = Client()

# Try GPT-4o first, fall back to Claude 3.5, then Gemini
fallback = FallbackChain(
    client=client,
    models=["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro"]
)

try:
    response = fallback.generate("Explain quantum computing")
    print(response.text)
except Exception as e:
    print(f"All models failed: {e}")
```

**Use Cases:**
- **Cost Optimization**: Try expensive model first, fall back to cheaper alternatives
- **Rate Limit Handling**: Automatically switch to other providers when rate limited
- **High Availability**: Ensure service continuity even if one provider is down
- **A/B Testing**: Test different models with automatic failover

**Async Support:**
```python
response = await fallback.generate_async("Your prompt here")
```

## Observability Middleware 🔭

### TracingMiddleware
Logs request lifecycles (start, end, error) with a trace ID.

```python
from aiclient import TracingMiddleware
import logging

logging.basicConfig(level=logging.INFO)
client.add_middleware(TracingMiddleware())
```

### OpenTelemetryMiddleware
Integrates with OpenTelemetry for distributed tracing (Datadog, Honeycomb, Jaeger).

```python
from aiclient import OpenTelemetryMiddleware
client.add_middleware(OpenTelemetryMiddleware(service_name="my-ai-service"))
```

### SemanticCacheMiddleware 🧠

Cache responses based on semantic similarity (embeddings) rather than exact text matching. Reduces costs by short-circuiting requests for similar prompts. Requires `numpy`.

```python
from aiclient.cache import SemanticCacheMiddleware
from typing import List
import httpx

# 1. Create an embedding provider (using OpenAI embeddings)
class OpenAIEmbedder:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        return response.json()["data"][0]["embedding"]

# 2. Add middleware to client
embedder = OpenAIEmbedder(api_key="sk-...")
client.add_middleware(SemanticCacheMiddleware(
    embedder=embedder,
    threshold=0.95  # 95% similarity required for cache hit
))

# 3. Similar prompts return cached results
response1 = client.chat("gpt-4o").generate("What is Python?")
response2 = client.chat("gpt-4o").generate("What's Python programming language?")  # Cache hit!
```

**Parameters:**
- `embedder`: Object with `embed(text: str) -> List[float]` method
- `threshold`: Cosine similarity threshold (0.0-1.0). Higher = stricter matching
- `backend`: Optional custom vector store (defaults to in-memory)

**Supported Embedders:**
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-english-v3.0`
- HuggingFace: `sentence-transformers/*`
