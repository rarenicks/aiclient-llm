import json
from typing import Any, Dict, Tuple, Optional
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage

class OpenAIProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def prepare_request(self, model: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        return "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        content = response_data["choices"][0]["message"]["content"]
        usage_data = response_data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return ModelResponse(
            text=content, 
            raw=response_data, 
            usage=usage,
            provider="openai"
        )

    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        raw_str = chunk.get("raw", "")
        if not raw_str or not raw_str.startswith("data: "):
            return None
        
        data_str = raw_str[6:].strip() # Remove "data: "
        if data_str == "[DONE]":
            return None
            
        try:
            data = json.loads(data_str)
            delta = data["choices"][0]["delta"].get("content", "")
            return StreamChunk(text=delta, delta=delta)
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
