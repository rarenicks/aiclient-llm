from typing import Optional, Dict
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

class Client:
    """
    Main entry point for aiclient.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        transport_factory: Optional[callable] = None
    ):
        self.keys = {
            "openai": openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "google": google_api_key or os.getenv("GEMINI_API_KEY"),
            "xai": xai_api_key or os.getenv("XAI_API_KEY"),
        }
        self.transport_factory = transport_factory or HTTPTransport

    def chat(self, model_name: str) -> ChatModel:
        """Create a chat model interface."""
        provider = self._resolve_provider(model_name)
        transport = self.transport_factory(
            base_url=provider.base_url,
            headers=provider.headers
        )
        return ChatModel(model_name, provider, transport)

    def _resolve_provider(self, model_name: str) -> Provider:
        if model_name.startswith("claude"):
            return AnthropicProvider(api_key=self.keys["anthropic"] or "")
        elif model_name.startswith("gemini"):
            return GoogleProvider(api_key=self.keys["google"] or "")
        elif "grok" in model_name:
            return OpenAIProvider(
                api_key=self.keys["xai"] or "",
                base_url="https://api.x.ai/v1"
            )
        else:
            # Default to OpenAI
            return OpenAIProvider(api_key=self.keys["openai"] or "")
