import contextvars
import logging
import re
from typing import Any, List, Protocol, Union

from .data_types import BaseMessage, ModelResponse

# ContextVar to store request-scoped model name
_request_model_context = contextvars.ContextVar("request_model", default=None)


class Middleware(Protocol):
    def before_request(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage], ModelResponse]:
        """
        Intercept and modify the request before it is sent to the provider.
        Returns the modified prompt (or messages).
        If a ModelResponse is returned, the provider call is skipped and this response
        is returned immediately (short-circuit).
        """
        ...

    def after_response(self, response: ModelResponse) -> ModelResponse:
        """
        Intercept and modify the response after it is received from the provider.
        Returns the modified response.
        """
        ...

    def on_error(self, error: Exception, model: str, **kwargs: Any) -> None:
        """
        Hook called when an error occurs during generation.
        kwargs may contain 'attempt' (int) and other context.
        """
        ...


class CostTrackingMiddleware:
    """
    Middleware to track total usage/cost across requests.
    Includes estimated USD pricing for common models.
    """

    # Pricing per 1M tokens (approximate, as of January 2026)
    PRICING = {
        # OpenAI - GPT-5 series
        "gpt-5.2": {"input": 1.75, "cache_read_input": 0.175, "output": 14.0},
        "gpt-5.2-pro": {"input": 21.0, "output": 168.0},
        "gpt-5.1": {"input": 1.25, "cache_read_input": 0.125, "output": 10.0},
        "gpt-5": {"input": 1.25, "cache_read_input": 0.125, "output": 10.0},
        "gpt-5-mini": {"input": 0.25, "cache_read_input": 0.025, "output": 2.0},
        "gpt-5-nano": {"input": 0.25, "cache_read_input": 0.025, "output": 2.0},
        # OpenAI - GPT-4o series
        "gpt-4o": {"input": 2.5, "cache_read_input": 1.25, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "cache_read_input": 0.075, "output": 1.6},
        "gpt-4o-audio": {"input": 2.5, "output": 10.0},
        # OpenAI - GPT-4 series
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        # OpenAI - GPT-3.5
        "gpt-3.5": {"input": 1.5, "output": 2},
        # OpenAI - o-series (reasoning)
        "o1": {"input": 15.0, "cache_read_input": 7.5, "output": 60.0},
        "o1-mini": {"input": 1.1, "cache_read_input": 0.55, "output": 4.4},
        "o3": {"input": 2.0, "cache_read_input": 0.5, "output": 8.0},
        "o3-mini": {"input": 1.1, "cache_read_input": 0.55, "output": 4.4},
        # Anthropic - Claude 4 / Opus 4.5 / Sonnet 4
        "claude-opus-4.5": {
            "input": 15.0,
            "cache_read_input": 0.5,
            "output": 25.0,
            "cache_write": 6.25,
        },
        "claude-opus-4": {
            "input": 15.0,
            "cache_read_input": 1.5,
            "output": 75.0,
            "cache_write": 18.75,
        },
        "claude-sonnet-4": {
            "input": 3.0,
            "cache_read_input": 0.3,
            "output": 15.0,
            "cache_write": 3.75,
        },
        # Anthropic - Claude 3.7
        "claude-3-7-sonnet": {
            "input": 3.0,
            "cache_read_input": 0.3,
            "output": 15.0,
            "cache_write": 3.75,
        },
        # Anthropic - Claude 3.5
        "claude-3-5-sonnet": {"input": 3.0, "cache_read_input": 0.3, "output": 15.0},
        "claude-3-5-haiku": {
            "input": 0.8,
            "cache_read_input": 0.08,
            "output": 4.0,
            "cache_write": 1.0,
        },
        # Anthropic - Claude 3
        "claude-3-opus": {
            "input": 15.0,
            "cache_read_input": 1.5,
            "output": 75.0,
            "cache_write": 18.75,
        },
        "claude-3-sonnet": {"input": 3.0, "cache_read_input": 0.3, "output": 15.0},
        "claude-3-haiku": {
            "input": 0.25,
            "cache_read_input": 0.025,
            "output": 1.25,
            "cache_write": 0.3,
        },
        # Google - Gemini 3
        "gemini-3-pro": {"input": 2.0, "cache_read_input": 0.2, "output": 12.0},
        "gemini-3-flash": {"input": 0.5, "cache_read_input": 0.05, "output": 3.0},
        # Google - Gemini 2.5
        "gemini-2.5-pro": {"input": 1.25, "cache_read_input": 0.125, "output": 10.0},
        "gemini-2.5-flash": {"input": 0.3, "cache_read_input": 0.03, "output": 2.5},
        # xAI - Grok 4
        "grok-4": {"input": 3.0, "cache_read_input": 0.76, "output": 15.0},
        "grok-4-1-fast": {"input": 0.2, "cache_read_input": 0.05, "output": 0.5},
        "grok-4-fast": {"input": 0.2, "cache_read_input": 0.05, "output": 0.5},
        # xAI - Grok 3
        "grok-3": {"input": 3.0, "cache_read_input": 0.6, "output": 15.0},
        "grok-3-fast": {"input": 5.0, "cache_read_input": 1.0, "output": 25.0},
        "grok-3-mini": {"input": 0.3, "cache_read_input": 0.06, "output": 0.5},
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_cache_read_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_input_tokens = 0
        self.total_cost_usd = 0.0

    def before_request(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        # Store model context if needed, but we can usually get it from response or
        # just assume for now we rely on the provider returning usage.
        # Issue: ModelResponse doesn't always contain the requested model name if
        # provider doesn't echo it.
        # But CostTracker is a stateful object, we can't easily map request->response
        # without a request ID.
        # For v0.1 simplicty, let's assume we can match loosely or we might need the
        # model name passed to after_response?
        # The Middleware protocol for `after_response` only takes response.
        # We might need to store the last requested model on the middleware instance?
        # NOT thread safe, but acceptable for this simple synchronous client.
        # Use contextvars for thread-safe/async-safe context storage
        _request_model_context.set(model)
        return prompt

    async def before_request_async(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        return self.before_request(model, prompt)

    def after_response(self, response: ModelResponse) -> ModelResponse:
        if response.usage:
            in_tok = response.usage.input_tokens
            cached_in_tok = response.usage.cache_read_input_tokens or 0
            out_tok = response.usage.output_tokens
            cache_creation_tok = response.usage.cache_creation_input_tokens or 0
            self.total_input_tokens += in_tok
            self.total_cache_read_input_tokens += cached_in_tok
            self.total_output_tokens += out_tok
            self.total_cache_creation_input_tokens += cache_creation_tok

            # pricing lookup
            # Retrieve model from context var
            model_name = _request_model_context.get()
            model_key = self._find_model_key(model_name)
            if model_key:
                rates = self.PRICING[model_key]
                cost = (
                    (in_tok - cached_in_tok) / 1_000_000 * rates["input"] +
                    cached_in_tok / 1_000_000 * rates.get("cache_read_input", 0) +
                    out_tok / 1_000_000 * rates["output"] +
                    cache_creation_tok / 1_000_000 * rates.get("cache_write", 0)
                )
                self.total_cost_usd += cost

        return response

    def on_error(self, error: Exception, model: str, **kwargs: Any) -> None:
        pass

    def _find_model_key(self, model_name: str) -> Union[str, None]:
        if not model_name:
            return None
        # Sort keys by length descending to match the most specific model first
        sorted_keys = sorted(self.PRICING.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in model_name:
                return key
        return None


class LoggingMiddleware:
    """
    Middleware to log requests and responses with optional key redaction.

    Logs request prompts and response text/usage to the specified logger.
    Automatically redacts common API key patterns from logged content.
    """

    # Patterns to redact from logs
    REDACT_PATTERNS = [
        (r"sk-[a-zA-Z0-9-]{20,}", "[REDACTED_OPENAI_KEY]"),
        (r"sk-ant-[a-zA-Z0-9-]{20,}", "[REDACTED_ANTHROPIC_KEY]"),
        (r"xai-[a-zA-Z0-9]{20,}", "[REDACTED_XAI_KEY]"),
        (r"AIza[a-zA-Z0-9_-]{35}", "[REDACTED_GOOGLE_KEY]"),
    ]

    def __init__(
        self,
        logger: logging.Logger = None,
        log_level: int = logging.INFO,
        log_prompts: bool = True,
        log_responses: bool = True,
        log_usage: bool = True,
        redact_keys: bool = True,
        max_prompt_length: int = 500,
        max_response_length: int = 500,
    ):
        """
        Initialize the logging middleware.

        Args:
            logger: Logger instance to use. Defaults to 'aiclient.requests'.
            log_level: Logging level (default: INFO).
            log_prompts: Whether to log request prompts.
            log_responses: Whether to log response text.
            log_usage: Whether to log token usage.
            redact_keys: Whether to redact API keys from logs.
            max_prompt_length: Max characters to log from prompt (0 = unlimited).
            max_response_length: Max characters to log from response (0 = unlimited).
        """
        self.logger = logger or logging.getLogger("aiclient.requests")
        self.log_level = log_level
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.log_usage = log_usage
        self.redact_keys = redact_keys
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def _redact(self, text: str) -> str:
        """Redact sensitive patterns from text."""
        if not self.redact_keys or not text:
            return text
        for pattern, replacement in self.REDACT_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if max_length <= 0 or len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def before_request(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        """Log the request before sending."""
        _request_model_context.set(model)

        if self.log_prompts:
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                # Extract text from messages
                prompt_text = " | ".join(
                    f"[{m.role}] "
                    f"{m.content if isinstance(m.content, str) else '[multimodal]'}"
                    for m in prompt
                )

            prompt_text = self._truncate(prompt_text, self.max_prompt_length)
            prompt_text = self._redact(prompt_text)

            self.logger.log(
                self.log_level, f"[REQUEST] model={model} prompt={prompt_text}"
            )

        return prompt

    async def before_request_async(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        return self.before_request(model, prompt)

    def after_response(self, response: ModelResponse) -> ModelResponse:
        """Log the response after receiving."""
        model_name = _request_model_context.get()
        log_parts = [f"[RESPONSE] model={model_name}"]

        if self.log_responses:
            response_text = self._truncate(response.text, self.max_response_length)
            response_text = self._redact(response_text)
            log_parts.append(f"text={response_text}")

        if self.log_usage and response.usage:
            log_parts.append(
                f"tokens={{in={response.usage.input_tokens}, "
                f"out={response.usage.output_tokens}, "
                f"total={response.usage.total_tokens}}}"
            )

        self.logger.log(self.log_level, " ".join(log_parts))
        return response

    def on_error(self, error: Exception, model: str, **kwargs: Any) -> None:
        """Log errors."""
        attempt = kwargs.get("attempt", 0)
        self.logger.error(f"[ERROR] model={model} attempt={attempt} error={error}")
