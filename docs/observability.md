# Observability & Monitoring 🔭

`aiclient-llm` provides comprehensive tools to understand what your AI application is doing, how much it costs, and where errors are occurring.

## 1. Cost Tracking

Track token usage and estimated USD costs in real-time. The tracker maintains an internal database of pricing for OpenAI, Anthropic, Google, and xAI models (updated Jan 2026).

```python
from aiclient import Client, CostTrackingMiddleware

client = Client()
tracker = CostTrackingMiddleware()
client.add_middleware(tracker)

# Run your app...
await client.chat("gpt-4o").generate_async("Hello")
await client.chat("claude-3-haiku").generate_async("Hi")

# Check usage
print(f"Total Cost: ${tracker.total_cost_usd:.4f}")
print(f"Input Tokens: {tracker.total_input_tokens}")
print(f"Output Tokens: {tracker.total_output_tokens}")

# Reset stats
tracker.reset()
```

## 2. Logging

Debug your application with detailed logs. Includes built-in support for redacting sensitive API keys.

```python
from aiclient import LoggingMiddleware

logger = LoggingMiddleware(
    log_prompts=True,     # Log input prompts
    log_responses=True,   # Log model outputs
    redact_keys=True,     # Replace API keys with 'sk-...'
    max_length=1000       # Truncate long logs
)
client.add_middleware(logger)
```

## 3. OpenTelemetry Integration

For production applications, `aiclient-llm` integrates with the OpenTelemetry (OTel) standard. This allows you to export traces to backend platforms like:
- Jaeger
- Zipkin
- Datadog
- Honeycomb
- Grafana Tempo

The `OpenTelemetryMiddleware` automatically creates spans for:
- HTTP Requests vs Cache Hits
- Token Usage
- Model Names
- Error events

```python
from aiclient import OpenTelemetryMiddleware

# Ensure you have 'opentelemetry-api' installed
otel = OpenTelemetryMiddleware(
    service_name="my-ai-service",
    tracer_provider=None # Uses global tracer by default
)
client.add_middleware(otel)
```
