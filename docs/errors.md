# Troubleshooting & Errors 🐛

When things go wrong, `aiclient` tries to tell you why with specific exception classes.

## Common Exceptions

| Exception | HTTP Code | Meaning |
|-----------|-----------|---------|
| `AuthenticationError` | 401, 403 | Invalid API key. Check your `.env` file. |
| `RateLimitError` | 429 | You're sending too many requests. Retry later. |
| `InvalidRequestError` | 400 | The provider rejected your request (e.g. invalid JSON, usage violations). |
| `ProviderError` | 5xx | The AI provider is down or experiencing issues. |
| `NetworkError` | N/A | Network failure (DNS, timeout). |

## Debug Mode

If you're unsure what's happening, enable debug mode to see raw request/response logs.
**Warning**: This logs your API keys and data to the console.

```python
from aiclient import Client

client = Client(debug=True)
# Logs will appear in stderr
```

You can also use standard Python logging configuration:

```python
import logging
logging.getLogger("aiclient").setLevel(logging.DEBUG)
```
