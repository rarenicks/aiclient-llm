import asyncio
import random
import time
import httpx
from typing import Optional, List, Union, Any
from ..data_types import BaseMessage, ModelResponse
from ..middleware import Middleware

class RetryMiddleware:
    """
    Middleware that retries requests on transient failures (429, 5xx).
    Implements exponential backoff with jitter.
    """
    def __init__(
        self, 
        max_retries: int = 3, 
        backoff_factor: float = 1.0, 
        max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def before_request(
        self, 
        model: str, 
        prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        # No Modification
        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        return response

    def on_error(self, error: Exception, model: str, **kwargs: Any) -> None:
        """
        Calculates backoff and sleeps if retryable. 
        If not retryable, raises error to abort the model's loop.
        """
        attempt = kwargs.get("attempt", 0)
        
        if not self.should_retry(error):
            # Not a retryable error (e.g. 400 Bad Request)
            # Raise it to break the ChatModel loop (which catches Exception)
            # Note: This raised exception will be caught by ChatModel's except block?
            # Wait, ChatModel: try...except Exception as e...
            # If mw.on_error raises E2, it bubbles out of the except block (chained).
            # So yes, raising here aborts the loop.
            raise error

        # If we are here, it IS retryable.
        # Check if we exceeded our own max_retries?
        # Only if we want to valid independent of ChatModel.
        if attempt >= self.max_retries:
             raise error
             
        # Sleep
        delay = self.calculate_delay(attempt)
        time.sleep(delay)
        
    async def on_error_async(self, error: Exception, model: str, **kwargs: Any) -> None:
        """Async version of on_error."""
        attempt = kwargs.get("attempt", 0)
        
        if not self.should_retry(error):
            raise error

        if attempt >= self.max_retries:
             raise error
             
        # Async Sleep
        delay = self.calculate_delay(attempt)
        await asyncio.sleep(delay)

    def should_retry(self, exception: Exception) -> bool:
        """Helper to determine if an exception is retryable."""
        if isinstance(exception, httpx.HTTPStatusError):
            code = exception.response.status_code
            return code == 429 or 500 <= code < 600
        return False
        
    def calculate_delay(self, attempt: int) -> float:
        delay = self.backoff_factor * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self.max_delay)

# Wait, looking at ChatModel.generate (viewed in step 2184 output trace):
# for attempt in range(self.max_retries + 1):
#    try: response = self.transport.send(...) ... break
#    except Exception as e: for mw in self.middlewares: mw.on_error(e) 
# 
# It loops, but it doesn't sleep! It just retries immediately?
# And it doesn't check IF it should retry based on error type.
# So `RetryMiddleware` is actually critical to inject strict retry *logic* (sleep + check).

# BUT `on_error` is a notification hook. It doesn't control the loop flow (continue vs raise).
# Users want `RetryMiddleware` to *handle* the retry.
# If `ChatModel` loop is dumb (always retries), then `RetryMiddleware.on_error` can enforce the sleep.
# If `ChatModel` loop retries on ANY exception, that's bad (e.g. 400 Bad Request).
# We might need to Modify `ChatModel` to delegate "should I continue?" to middleware?
# Or `RetryMiddleware` sleeps in `on_error`, and raises if it decides we shouldn't retry.

# Plan:
# 1. Implement `on_error` to checking exception.
# 2. If retryable: sleep for backoff.
# 3. If NOT retryable: raise the exception (breaking the model's dumb loop).
# 4. We need to know which attempt we are on? `ChatModel` doesn't pass attempt count to `on_error`.

# This implies `RetryMiddleware` needs state or `ChatModel` needs update.
# Simpler approach for v0.3: 
# Implement `RetryMiddleware` as a WRAPPER around `model.generate`? No, middleware is injected.
# Best approach: Update `ChatModel` logic to cooperate with `RetryMiddleware` or improve the loop.
# Given constraints, I'll attempt this:
# `ChatModel` loop is: try...except.
# `RetryMiddleware.on_error(e)`:
#   if not retryable(e): raise e
#   else: sleep(backoff)
# This works if `on_error` can raise.

# Limitation: We don't know "current attempt" for exponential backoff inside `on_error` easily unless we track it?
# Or `ChatModel` should be updated to do the sleep.

# Let's rely on `ChatModel`'s loop for the *retry mechanism* but improve `on_error` to handle the *delay* and *filter*.
# I will implement `on_error` to blocking-sleep (valid for sync). For async, we need `on_error_async`.

# Let's verify Middleware protocol first.
pass
