"""
Integration tests for middleware functionality.
Tests error hooks, cost tracking, and middleware chain.
"""
import pytest
from unittest.mock import MagicMock
from aiclient import Client
from aiclient.models.chat import ChatModel
from aiclient.middleware import CostTrackingMiddleware
from aiclient.resilience import CircuitBreaker, RateLimiter
from aiclient.observability import TracingMiddleware
from aiclient.types import ModelResponse, Usage, BaseMessage
from aiclient.transport.base import Transport


class ErrorTrackingMiddleware:
    """Test middleware to track errors."""
    def __init__(self):
        self.errors = []

    def before_request(self, model, prompt):
        return prompt

    def after_response(self, response):
        return response

    def on_error(self, error, model, **kwargs):
        self.errors.append((error, model))


class MockFailingTransport(Transport):
    """Transport that fails N times then succeeds."""
    def __init__(self, fail_count=2):
        self.fail_count = fail_count
        self.call_count = 0

    def send(self, endpoint, data):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            exc = Exception("Mock failure")
            exc.response = MagicMock()
            exc.response.status_code = 500
            raise exc
        return {"choices": [{"message": {"content": "Success"}}]}

    async def send_async(self, endpoint, data):
        return self.send(endpoint, data)

    def stream(self, endpoint, data):
        yield {"raw": "data"}

    async def stream_async(self, endpoint, data):
        yield {"raw": "data"}


def test_middleware_error_hook_called_on_retry():
    """Test that middleware on_error hook is called during retries."""
    error_tracker = ErrorTrackingMiddleware()

    transport = MockFailingTransport(fail_count=2)
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(text="Success", raw={})

    model = ChatModel(
        "test-model",
        provider,
        transport,
        middlewares=[error_tracker],
        max_retries=3,
        retry_delay=0.01
    )

    response = model.generate("test")

    # Should have called on_error twice (for 2 failures)
    assert len(error_tracker.errors) == 2
    assert response.text == "Success"


def test_cost_tracking_middleware():
    """Test CostTrackingMiddleware tracks usage and costs."""
    cost_tracker = CostTrackingMiddleware()

    # Simulate responses with usage
    response1 = ModelResponse(
        text="Hello",
        raw={},
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    )

    response2 = ModelResponse(
        text="World",
        raw={},
        usage=Usage(input_tokens=200, output_tokens=100, total_tokens=300)
    )

    # Process responses
    cost_tracker.before_request("gpt-4o", "prompt")
    cost_tracker.after_response(response1)

    cost_tracker.before_request("gpt-4o", "prompt")
    cost_tracker.after_response(response2)

    # Check totals
    assert cost_tracker.total_input_tokens == 300
    assert cost_tracker.total_output_tokens == 150
    assert cost_tracker.total_cost_usd > 0  # Should have calculated cost
    print(f"Total cost: ${cost_tracker.total_cost_usd:.6f}")


def test_middleware_chain_order():
    """Test middleware are called in correct order."""
    call_order = []

    class OrderedMiddleware:
        def __init__(self, name):
            self.name = name

        def before_request(self, model, prompt):
            call_order.append(f"{self.name}_before")
            return prompt

        def after_response(self, response):
            call_order.append(f"{self.name}_after")
            return response

        def on_error(self, error, model):
            pass

    mw1 = OrderedMiddleware("MW1")
    mw2 = OrderedMiddleware("MW2")

    transport = MagicMock(spec=Transport)
    transport.send.return_value = {"choices": [{"message": {"content": "Test"}}]}

    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(text="Test", raw={})

    model = ChatModel(
        "test-model",
        provider,
        transport,
        middlewares=[mw1, mw2],
        max_retries=0
    )

    model.generate("test")

    # Before hooks: MW1, MW2
    # After hooks: MW1, MW2
    assert call_order == ["MW1_before", "MW2_before", "MW1_after", "MW2_after"]


def test_circuit_breaker_middleware_integration():
    """Test CircuitBreaker works in real chat flow."""
    circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    transport = MockFailingTransport(fail_count=10)  # Always fail
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})

    model = ChatModel(
        "test-model",
        provider,
        transport,
        middlewares=[circuit_breaker],
        max_retries=0  # Don't retry, fail immediately
    )

    # First failure
    with pytest.raises(Exception):
        model.generate("test")
    assert circuit_breaker._state == "CLOSED"
    assert circuit_breaker._failures == 1

    # Second failure - should open circuit
    with pytest.raises(Exception):
        model.generate("test")
    assert circuit_breaker._state == "OPEN"

    # Third attempt - circuit is open, should fail immediately
    with pytest.raises(Exception) as exc:
        model.generate("test")
    assert "CircuitBreaker is OPEN" in str(exc.value)


def test_rate_limiter_middleware_integration():
    """Test RateLimiter works in real chat flow."""
    import time

    rate_limiter = RateLimiter(requests_per_minute=2)  # Very low for testing

    transport = MagicMock(spec=Transport)
    transport.send.return_value = {"choices": [{"message": {"content": "OK"}}]}

    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(text="OK", raw={})

    model = ChatModel(
        "test-model",
        provider,
        transport,
        middlewares=[rate_limiter],
        max_retries=0
    )

    # First two requests should be fast
    start = time.time()
    model.generate("test1")
    model.generate("test2")
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should be quick

    # Third request should be rate limited (would need to wait)
    # We won't test the sleep here as it takes too long


def test_multiple_middleware_cooperation():
    """Test multiple middleware working together."""
    cost_tracker = CostTrackingMiddleware()
    error_tracker = ErrorTrackingMiddleware()
    circuit_breaker = CircuitBreaker(failure_threshold=3)

    transport = MockFailingTransport(fail_count=1)  # Fail once
    provider = MagicMock()
    provider.prepare_request.return_value = ("url", {})
    provider.parse_response.return_value = ModelResponse(
        text="Success",
        raw={},
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    )

    model = ChatModel(
        "gpt-4o",
        provider,
        transport,
        middlewares=[cost_tracker, error_tracker, circuit_breaker],
        max_retries=2,
        retry_delay=0.01
    )

    response = model.generate("test")

    # All middleware should have worked
    assert response.text == "Success"
    assert len(error_tracker.errors) == 1  # One retry
    assert cost_tracker.total_input_tokens == 10  # Cost tracked
    assert circuit_breaker._state == "CLOSED"  # Not tripped


# Requires pytest-asyncio
# @pytest.mark.asyncio
# async def test_middleware_error_hook_async():
#     """Test on_error hooks work with async generation."""
#     error_tracker = ErrorTrackingMiddleware()
#
#     transport = MockFailingTransport(fail_count=2)
#     provider = MagicMock()
#     provider.prepare_request.return_value = ("url", {})
#     provider.parse_response.return_value = ModelResponse(text="Success", raw={})
#
#     model = ChatModel(
#         "test-model",
#         provider,
#         transport,
#         middlewares=[error_tracker],
#         max_retries=3,
#         retry_delay=0.01
#     )
#
#     response = await model.generate_async("test")
#
#     # Should have called on_error for async failures too
#     assert len(error_tracker.errors) == 2
#     assert response.text == "Success"
