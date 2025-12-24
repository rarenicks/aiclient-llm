from typing import Protocol, Union, List, Any
from .data_types import BaseMessage, ModelResponse, Usage

class Middleware(Protocol):
    def before_request(self, model: str, prompt: Union[str, List[BaseMessage]]) -> Union[str, List[BaseMessage], ModelResponse]:
        """
        Intercept and modify the request before it is sent to the provider.
        Returns the modified prompt (or messages).
        If a ModelResponse is returned, the provider call is skipped and this response is returned immediately (short-circuit).
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
    # Pricing per 1M tokens (approximate, as of late 2024/early 2025)
    PRICING = {
        # OpenAI
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-3.5": {"input": 0.5, "output": 1.5},
        # Anthropic
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        # Google (Gemini 1.5/2.0 often has free tier or low pricing, using 1.5 Pro proxy)
        "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
        "gemini-1.5-flash": {"input": 0.35, "output": 0.7},
        "gemini-2.0": {"input": 0.0, "output": 0.0}, # Often free in preview
        # xAI
        "grok-2": {"input": 2.0, "output": 10.0}, # Estimated/Proxy
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def before_request(self, model: str, prompt: Union[str, List[BaseMessage]]) -> Union[str, List[BaseMessage]]:
        # Store model context if needed, but we can usually get it from response or just assume 
        # for now we rely on the provider returning usage.
        # Issue: ModelResponse doesn't always contain the requested model name if provider doesn't echo it.
        # But CostTracker is a stateful object, we can't easily map request->response without a request ID.
        # For v0.1 simplicty, let's assume we can match loosely or we might need the model name passed to after_response?
        # The Middleware protocol for `after_response` only takes response.
        # We might need to store the last requested model on the middleware instance? 
        # NOT thread safe, but acceptable for this simple synchronous client.
        self._last_model = model
        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        if response.usage:
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            
            # pricing lookup
            model_key = self._find_model_key(self._last_model)
            if model_key:
                rates = self.PRICING[model_key]
                cost = (in_tok / 1_000_000 * rates["input"]) + (out_tok / 1_000_000 * rates["output"])
                self.total_cost_usd += cost

        return response

    def on_error(self, error: Exception, model: str, **kwargs: Any) -> None:
        pass

    def _find_model_key(self, model_name: str) -> Union[str, None]:
        if not model_name:
            return None
        for key in self.PRICING:
            if key in model_name:
                return key
        return None
