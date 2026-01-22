import threading
import time
from typing import List, Union

from ..data_types import BaseMessage, ModelResponse
from ..middleware import Middleware
from .retries import RetryMiddleware


class CircuitBreaker(Middleware):
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._failures = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    def before_request(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        with self._lock:
            current_time = time.time()

            if self._state == "OPEN":
                if current_time - self._last_failure_time > self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    # Allow this request to proceed as probe
                else:
                    raise Exception(
                        f"CircuitBreaker is OPEN for {model}. Too many failures."
                    )

        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        # If we successfully got a response, reset
        with self._lock:
            if self._state == "HALF_OPEN":
                self._state = "CLOSED"
                self._failures = 0
            elif self._state == "CLOSED":
                self._failures = 0
        return response

    def on_error(self, error: Exception, model: str, **kwargs) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"


class RateLimiter(Middleware):
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = None):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.window = 60.0
        self._request_timestamps = []
        self._token_timestamps = [] # List of (timestamp, count)
        self._lock = threading.Lock()

    def before_request(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        token_count = self._estimate_tokens(prompt)
        
        with self._lock:
            now = time.time()
            self._cleanup(now)
            
            wait_time = max(
                self._check_rpm(now),
                self._check_tpm(now, token_count)
            )
            
            if wait_time > 0:
                time.sleep(wait_time)
                # Re-check or just assume we drift slightly. 
                # For strictness we might loop, but simple sleep is okay.
                now = time.time() # Update now after sleep
                self._cleanup(now) # Cleanup again if needed

            self._record_usage(now, token_count)

        return prompt

    async def before_request_async(
        self, model: str, prompt: Union[str, List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        token_count = self._estimate_tokens(prompt)
        
        # Lock not strictly safe if mixed with threaded sync calls, 
        # but for async-only typically single threaded. 
        # However, we should be careful. 
        # Ideally we use an async lock or rely on the sync lock wrapping fast ops.
        # Since _cleanup/_check are fast non-blocking, we can use the sync lock.
        # But we MUST NOT sleep holding the lock.
        
        with self._lock:
            now = time.time()
            self._cleanup(now)
            wait_time = max(
                self._check_rpm(now),
                self._check_tpm(now, token_count)
            )
        
        if wait_time > 0:
            import asyncio
            await asyncio.sleep(wait_time)
            with self._lock:
                 now = time.time()
                 self._cleanup(now)

        with self._lock:
             self._record_usage(time.time(), token_count)

        return prompt

    def _estimate_tokens(self, prompt: Union[str, List[BaseMessage]]) -> int:
        if isinstance(prompt, str):
            return len(prompt) // 4
        
        # Estimate for list of messages
        text = ""
        for m in prompt:
            text += str(m.content)
        return len(text) // 4

    def _cleanup(self, now: float):
        self._request_timestamps = [t for t in self._request_timestamps if now - t < self.window]
        self._token_timestamps = [t for t in self._token_timestamps if now - t[0] < self.window]

    def _check_rpm(self, now: float) -> float:
        if len(self._request_timestamps) >= self.rpm:
             return self.window - (now - self._request_timestamps[0])
        return 0.0

    def _check_tpm(self, now: float, cost: int) -> float:
        if not self.tpm:
            return 0.0
        
        current_tokens = sum(t[1] for t in self._token_timestamps)
        if current_tokens + cost > self.tpm:
            # Simple heuristic: wait until enough tokens expire.
            # This is tricky with sliding window. 
            # We find how many tokens we need to free up.
            needed = (current_tokens + cost) - self.tpm
            
            freed = 0
            for ts, count in self._token_timestamps:
                freed += count
                if freed >= needed:
                    return self.window - (now - ts)
            
            # If we need more than entire history? Wait full window?
            return self.window
        return 0.0

    def _record_usage(self, now: float, tokens: int):
        self._request_timestamps.append(now)
        if self.tpm:
            self._token_timestamps.append((now, tokens))

    def after_response(self, response: ModelResponse) -> ModelResponse:
        return response

    def on_error(self, error: Exception, model: str, **kwargs) -> None:
        pass


class FallbackChain:
    """
    Executes a prompt across a list of models, falling back to the next on failure.
    """

    def __init__(self, client, models: List[str]):
        self.client = client
        self.models = models

    def generate(
        self, prompt: Union[str, List[BaseMessage]], **kwargs
    ) -> ModelResponse:
        last_exception = None
        for model in self.models:
            try:
                return self.client.chat(model).generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                # Continue to next model
                continue
        raise last_exception or Exception("All fallback models failed")

    async def generate_async(
        self, prompt: Union[str, List[BaseMessage]], **kwargs
    ) -> ModelResponse:
        last_exception = None
        for model in self.models:
            try:
                return await self.client.chat(model).generate_async(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                continue
        raise last_exception or Exception("All fallback models failed")


class LoadBalancer:
    """
    Distributes requests across multiple models/endpoints using Round Robin.
    """

    def __init__(self, client, models: List[str]):
        self.client = client
        self.models = models
        self._index = 0
        self._lock = threading.Lock()

    def _get_next_model(self) -> str:
        with self._lock:
            model = self.models[self._index]
            self._index = (self._index + 1) % len(self.models)
            return model

    def generate(
        self, prompt: Union[str, List[BaseMessage]], **kwargs
    ) -> ModelResponse:
        model = self._get_next_model()
        return self.client.chat(model).generate(prompt, **kwargs)

    async def generate_async(
        self, prompt: Union[str, List[BaseMessage]], **kwargs
    ) -> ModelResponse:
        model = self._get_next_model()
        return await self.client.chat(model).generate_async(prompt, **kwargs)


__all__ = [
    "CircuitBreaker",
    "RateLimiter",
    "FallbackChain",
    "LoadBalancer",
    "RetryMiddleware",
]
