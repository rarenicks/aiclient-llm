import logging
import os
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from .middleware import Middleware
from .models.chat import ChatModel
from .providers.anthropic import AnthropicProvider
from .providers.base import Provider
from .providers.google import GoogleProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .transport.http import HTTPTransport

load_dotenv()

# Default prefix-to-provider routing
MODEL_PREFIX_MAP = {
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "grok-": "xai",
}

# Model lists for each provider (updated January 2026)
OPENAI_MODELS = [
    # GPT-4o series
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-12-17",
    # GPT-4 series
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-4-0613",
    # GPT-3.5
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    # o-series (reasoning models)
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # Embeddings
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

ANTHROPIC_MODELS = [
    # Claude 4 / Opus 4.5 series (latest)
    "claude-opus-4-20250514",
    "claude-opus-4-5-20250514",
    "claude-sonnet-4-20250514",
    # Claude 3.7 series
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest",
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 series
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

GEMINI_MODELS = [
    # Gemini 3 series (latest)
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    # Gemini 2.5 series
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    # Gemini 2.0 series
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    # Gemini 1.5 series (legacy)
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    # Embeddings
    "text-embedding-004",
]

XAI_MODELS = [
    # Grok 3 series (latest)
    "grok-3",
    "grok-3-latest",
    "grok-3-fast",
    "grok-3-fast-latest",
    "grok-3-mini",
    "grok-3-mini-latest",
    "grok-3-mini-fast",
    "grok-3-mini-fast-latest",
    # Grok 2 series
    "grok-2",
    "grok-2-latest",
    "grok-2-1212",
    "grok-2-vision",
    "grok-2-vision-latest",
    "grok-2-vision-1212",
    # Grok beta
    "grok-beta",
    "grok-vision-beta",
    # Embeddings
    "grok-embedding-beta",
]


class Client:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        transport_factory=None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        google_api_version: str = "v1beta",
        debug: bool = False,
    ):
        self.keys = {
            "openai": openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "google": (google_api_key or os.getenv("GEMINI_API_KEY") or
                os.getenv("GOOGLE_API_KEY")),
            "xai": xai_api_key or os.getenv("XAI_API_KEY"),
        }
        self.ollama_base_url = ollama_base_url
        self.google_api_version = google_api_version
        self.transport_factory = transport_factory or HTTPTransport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._middlewares: List[Middleware] = []

        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            # Ensure our logger is set to DEBUG even if root wasn't overridden
            logging.getLogger("aiclient").setLevel(logging.DEBUG)

    def add_middleware(self, middleware: Middleware):
        """Register a middleware to the pipeline."""
        self._middlewares.append(middleware)

    def _get_provider(self, model: str) -> Tuple[Provider, str]:
        # Check for explicit provider:model syntax
        if ":" in model:
            provider_prefix, real_model = model.split(":", 1)
            if provider_prefix == "ollama":
                # Use configured base_url or default
                return OllamaProvider(
                    base_url=self.ollama_base_url or "http://localhost:11434/v1"
                ), real_model
            elif provider_prefix == "openai":
                return OpenAIProvider(api_key=self.keys["openai"]), real_model
            elif provider_prefix == "anthropic":
                return AnthropicProvider(api_key=self.keys["anthropic"]), real_model
            elif provider_prefix == "google":
                return GoogleProvider(
                    api_key=self.keys["google"], api_version=self.google_api_version
                ), real_model
            elif provider_prefix == "xai":
                return OpenAIProvider(
                    api_key=self.keys["xai"], base_url="https://api.x.ai/v1"
                ), real_model

        # Fallback to legacy prefix matching (keep for backward compatibility)
        # Fallback to prefix matching
        for prefix, provider_key in MODEL_PREFIX_MAP.items():
            if model.startswith(prefix):
                if provider_key == "openai":
                    return OpenAIProvider(api_key=self.keys["openai"]), model
                elif provider_key == "anthropic":
                    return AnthropicProvider(api_key=self.keys["anthropic"]), model
                elif provider_key == "google":
                    return GoogleProvider(
                        api_key=self.keys["google"], api_version=self.google_api_version
                    ), model
                elif provider_key == "xai":
                    return OpenAIProvider(
                        api_key=self.keys["xai"], base_url="https://api.x.ai/v1"
                    ), model

        # Exact match or default fallback or error
        if (
            model == "o1" or model == "o3"
        ):  # Special case for base reasoning models without hyphen
            return OpenAIProvider(api_key=self.keys["openai"]), model

        raise ValueError(
            f"Unknown model provider for {model}. "
            "Try using 'provider:model_name' syntax (e.g. 'ollama:llama3')."
        )

    def chat(self, model_name: str) -> ChatModel:
        provider, real_model_name = self._get_provider(model_name)
        transport = self.transport_factory(
            base_url=provider.base_url, headers=provider.headers, timeout=self.timeout
        )
        return ChatModel(
            real_model_name,
            provider,
            transport,
            self._middlewares,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

    async def embed(
        self, input: Union[str, List[str]], model: str
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the input text.
        """
        provider, real_model_name = self._get_provider(model)
        # Note: We are creating a fresh transport here.
        # In a real app we might want to cache transports or use a connection pool.
        transport = self.transport_factory(
            base_url=provider.base_url, headers=provider.headers
        )

        endpoint, data = provider.prepare_embeddings_request(real_model_name, input)
        response_data = await transport.send_async(endpoint, data)
        result = provider.parse_embeddings_response(response_data)

        if isinstance(input, str) and len(result) > 0:
            return result[0]
        return result

    async def embed_batch(self, inputs: List[str], model: str) -> List[List[float]]:
        """
        Generate embeddings for a batch of inputs.
        """
        return await self.embed(inputs, model)

    async def batch(
        self,
        inputs: List[Any],
        func: Callable[[Any], Coroutine[Any, Any, Any]],
        concurrency: int = 5,
        return_exceptions: bool = True,
    ) -> List[Any]:
        """
        Execute a function concurrently for a list of inputs.

        Args:
            inputs: List of inputs to process.
            func: Async function that accepts a single input.
            concurrency: Max parallel requests (default 5).
            return_exceptions: If True, returns Exception objects for failed
                              items instead of raising.

        Returns:
            List of results matching the order of inputs.
        """
        from .batch import BatchProcessor

        processor = BatchProcessor(concurrency=concurrency)
        return await processor.process(
            inputs, func, return_exceptions=return_exceptions
        )

    # Async context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close any open connections."""
        await self.close()

    async def close(self):
        """Close all open HTTP connections."""
        # Note: Individual transports are created per-request currently
        # This is here for future connection pooling support
        pass

    def list_models(self, provider: str = None) -> Dict[str, List[str]]:
        """
        List available models for the specified provider(s).

        Args:
            provider: Optional provider name ('openai', 'anthropic', 'google', 'xai').
                     If None, returns all providers.

        Returns:
            Dictionary mapping provider names to lists of model names.
        """
        all_models = {
            "openai": OPENAI_MODELS,
            "anthropic": ANTHROPIC_MODELS,
            "google": GEMINI_MODELS,
            "xai": XAI_MODELS,
        }

        if provider:
            provider = provider.lower()
            if provider not in all_models:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Available: {list(all_models.keys())}"
                )
            return {provider: all_models[provider]}

        return all_models

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """
        Count the number of tokens in a text string.

        Uses tiktoken for OpenAI models. For other providers, uses an approximation
        based on the OpenAI tokenizer (cl100k_base).

        Args:
            text: The text to count tokens for.
            model: Model name to use for tokenization (default: gpt-4o).

        Returns:
            Number of tokens in the text.
        """
        try:
            import tiktoken

            # Map model to encoding
            if (
                model.startswith("gpt-4")
                or model.startswith("gpt-3.5")
                or model.startswith("o1")
                or model.startswith("o3")
            ):
                encoding = tiktoken.encoding_for_model(model)
            else:
                # Use cl100k_base as default for other models (Claude, Gemini, etc.)
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except ImportError:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install with: pip install tiktoken"
            )
