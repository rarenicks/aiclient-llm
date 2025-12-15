import json
from typing import Any, Dict, Tuple, Optional
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage

class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def prepare_request(self, model: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        return "/messages", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        }

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        content = response_data["content"][0]["text"]
        usage_data = response_data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        return ModelResponse(
            text=content, 
            raw=response_data,
            usage=usage,
            provider="anthropic"
        )
        
    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        raw_str = chunk.get("raw", "")
        # Basic SSE parsing for Anthropic
        if not raw_str.startswith("data: "):
             return None
             
        data_str = raw_str[6:].strip()
        try:
            data = json.loads(data_str)
            if data["type"] == "content_block_delta":
                 delta = data["delta"]["text"]
                 return StreamChunk(text=delta, delta=delta)
        except (json.JSONDecodeError, KeyError):
            pass
        return None
