from .client import Client
from .agent import Agent
from .tools.base import Tool
from .providers.ollama import OllamaProvider
from .types import (
    UserMessage, SystemMessage, AssistantMessage, ToolMessage,
    Text, Image, ModelResponse, StreamChunk, Usage
)

from .middleware import Middleware, CostTrackingMiddleware
from .resilience import CircuitBreaker, RateLimiter, FallbackChain, LoadBalancer, RetryMiddleware
from .observability import TracingMiddleware, OpenTelemetryMiddleware
from .cache import SemanticCacheMiddleware
from .memory import ConversationMemory, SlidingWindowMemory
from .batch import BatchProcessor
from .testing import MockProvider, MockTransport
from .exceptions import (
    AIClientError, AuthenticationError, RateLimitError,
    ProviderError, InvalidRequestError, NetworkError
)

__version__ = "0.2.0"

__all__ = [
    "Client",
    "Agent",
    "Tool",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "Text",
    "Image",
    "ModelResponse",
    "StreamChunk",
    "Usage",
    "Middleware",
    "CostTrackingMiddleware",
    "CircuitBreaker",
    "RateLimiter",
    "RetryMiddleware",
    "FallbackChain",
    "LoadBalancer",
    "TracingMiddleware",
    "OpenTelemetryMiddleware",
    "SemanticCacheMiddleware",
    "ConversationMemory",
    "SlidingWindowMemory",
    "BatchProcessor",
    "MockProvider",
    "MockTransport",
    "AIClientError",
    "AuthenticationError",
    "RateLimitError",
    "ProviderError",
    "InvalidRequestError",
    "NetworkError",
]