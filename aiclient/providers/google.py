import json
from typing import Any, Dict, Tuple, Optional
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage

class GoogleProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def prepare_request(self, model: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        # Google API puts key in query param usually, but we want to stick to clean abstract transport.
        # However, basic HTTP transport can append key to URL if needed in transport layer or here.
        # For this design, let's assume we pass key as query param in endpoint.
        endpoint = f"/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        return endpoint, payload

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        try:
             content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
             content = ""
             
        meta = response_data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=meta.get("promptTokenCount", 0),
            output_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        )

        return ModelResponse(
            text=content, 
            raw=response_data,
            usage=usage,
            provider="google"
        )

    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        # Google streaming format is a bit different (array of JSON objects)
        # Placeholder for complex stream parsing which might differ from basic SSE
        return None 
