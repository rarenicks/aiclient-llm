from typing import Optional, Dict, List
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

class Client:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 xai_api_key: Optional[str] = None,
                 transport_factory=None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        
        self.keys = {
            "openai": openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "google": google_api_key or os.getenv("GEMINI_API_KEY"),
            "xai": xai_api_key or os.getenv("XAI_API_KEY"),
        }
        self.transport_factory = transport_factory or HTTPTransport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._middlewares: List[Middleware] = []

    def add_middleware(self, middleware: Middleware):
        """Register a middleware to the pipeline."""
        self._middlewares.append(middleware)

    def _get_provider(self, model: str) -> Provider:
        if model.startswith("gpt") or model.startswith("o1"):
            return OpenAIProvider(api_key=self.keys["openai"])
        elif model.startswith("grok"):
            return OpenAIProvider(api_key=self.keys["xai"], base_url="https://api.x.ai/v1")
        elif model.startswith("claude"):
            return AnthropicProvider(api_key=self.keys["anthropic"])
        elif model.startswith("gemini"):
            return GoogleProvider(api_key=self.keys["google"])
        else:
             # Default fallback or error
             raise ValueError(f"Unknown model provider for {model}")

    def chat(self, model_name: str) -> ChatModel:
        provider = self._get_provider(model_name)
        transport = self.transport_factory(
            base_url=provider.base_url,
            headers=provider.headers
        )
        return ChatModel(
            model_name, 
            provider, 
            transport, 
            self._middlewares,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
