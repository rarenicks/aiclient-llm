# Production Resilience 🛡️

In production environments, external APIs fail. They time out, rate limit you, or return 500 errors. `aiclient-llm` includes a robust set of resilience patterns to ensure your application stays up even when providers go down.

## Middleware Pipeline

Resilience features are implemented as **Middleware**. You can stack them in any order, but we recommend:
1. Retry (closest to network)
2. Circuit Breaker
3. Rate Limiter
4. Cache (closest to application)

## Features

### 1. Automatic Retries

Automatically retry failed requests with exponential backoff and jitter.

It handles:
- `429 Too Many Requests`
- `5xx Server Errors`
- Network timeouts

```python
from aiclient import Client, RetryMiddleware

# Configure internally in Client
client = Client(
    max_retries=3,
    retry_delay=1.0  # seconds
)

# OR add middleware manually for more control
client.add_middleware(RetryMiddleware(
    max_retries=5,
    retry_delay=0.5,
    backoff_factor=2.0,  # fail? wait 0.5s -> 1.0s -> 2.0s -> 4.0s
    jitter=True          # Randomize delay to prevent thundering herd
))
```

### 2. Circuit Breakers

Stop hitting a failing provider to prevent cascade failures and save latency. If a provider fails `N` times consecutively, the circuit "opens" and instantly fails subsequent requests for `recovery_timeout` seconds.

```python
from aiclient import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,   # Open after 5 consecutive failures
    recovery_timeout=60.0, # Wait 60s before trying again (Half-Open)
    monitor_exceptions=[   # Only break on these errors
        "NetworkError", 
        "TimeoutError", 
        "server_error"     # 5xx
    ]
)
client.add_middleware(cb)
```

### 3. Rate Limiters

Respect provider API limits (e.g., "60 RPM"). This runs locally to prevent you from getting 429s.

```python
from aiclient import RateLimiter

# Global rate limit
rl = RateLimiter(requests_per_minute=60)
client.add_middleware(rl)

# You can also scope limiters per provider if you instantitate separate clients
```

### 4. Fallback Chains

Ensure high availability by automatically failing over to alternative models if the primary one fails.

```python
from aiclient import FallbackChain

# Configure a primary client
client = Client()

# Create a fallback wrapper
chain = FallbackChain(client, [
    "gpt-4o",           # 1. Try Primary
    "claude-3-opus",    # 2. Try Anthropic if OpenAI fails
    "gemini-1.5-pro"    # 3. Try Gemini if both fail
])

# Use just like a normal model
try:
    response = chain.generate("Crucial business query")
    print(f"Success! Answered by: {response.model}")
except Exception:
    print("All models failed.")
```
