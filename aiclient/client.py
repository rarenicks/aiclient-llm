import logging
import asyncio

from typing import Optional, Dict, List, Tuple, Any, Callable, Coroutine
import os
from dotenv import load_dotenv

load_dotenv()

from .transport.base import Transport
from .transport.http import HTTPTransport
from .models.chat import ChatModel
from .providers.base import Provider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .middleware import Middleware

from .providers.ollama import OllamaProvider

class Client:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 xai_api_key: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 transport_factory=None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 debug: bool = False):
        
        self.keys = {
            "openai": openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "google": google_api_key or os.getenv("GEMINI_API_KEY"),
            "xai": xai_api_key or os.getenv("XAI_API_KEY"),
        }
        self.ollama_base_url = ollama_base_url
        self.transport_factory = transport_factory or HTTPTransport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._middlewares: List[Middleware] = []

        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            # Ensure our logger is set to DEBUG even if root wasn't overridden essentially
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
                return OllamaProvider(base_url=self.ollama_base_url or "http://localhost:11434/v1"), real_model
            elif provider_prefix == "openai":
                return OpenAIProvider(api_key=self.keys["openai"]), real_model
            elif provider_prefix == "anthropic":
                return AnthropicProvider(api_key=self.keys["anthropic"]), real_model
            elif provider_prefix == "google":
                return GoogleProvider(api_key=self.keys["google"]), real_model
            elif provider_prefix == "xai":
                 return OpenAIProvider(api_key=self.keys["xai"], base_url="https://api.x.ai/v1"), real_model
        
        # Fallback to legacy prefix matching (keep for backward compatibility)
        if model.startswith("gpt") or model.startswith("o1"):
            return OpenAIProvider(api_key=self.keys["openai"]), model
        elif model.startswith("grok"):
            return OpenAIProvider(api_key=self.keys["xai"], base_url="https://api.x.ai/v1"), model
        elif model.startswith("claude"):
            return AnthropicProvider(api_key=self.keys["anthropic"]), model
        elif model.startswith("gemini"):
            return GoogleProvider(api_key=self.keys["google"]), model
        else:
             # Default fallback or error
             raise ValueError(f"Unknown model provider for {model}. Try using 'provider:model_name' syntax (e.g. 'ollama:llama3').")

    def chat(self, model_name: str) -> ChatModel:
        provider, real_model_name = self._get_provider(model_name)
        transport = self.transport_factory(
            base_url=provider.base_url,
            headers=provider.headers
        )
        return ChatModel(
            real_model_name, 
            provider, 
            transport, 
            self._middlewares,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )

    async def batch(self, 
                    inputs: List[Any], 
                    func: Callable[[Any], Coroutine[Any, Any, Any]],
                    concurrency: int = 5,
                    return_exceptions: bool = True) -> List[Any]:
        """
        Execute a function concurrently for a list of inputs.
        
        Args:
            inputs: List of inputs to process.
            func: Async function that accepts a single input.
            concurrency: Max parallel requests (default 5).
            return_exceptions: If True, returns Exception objects for failed items instead of raising.
            
        Returns:
            List of results matching the order of inputs.
        """
        from .batch import BatchProcessor
        processor = BatchProcessor(concurrency=concurrency)
        return await processor.process(inputs, func, return_exceptions=return_exceptions)
