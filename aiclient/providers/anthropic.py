import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage

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

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False) -> Tuple[str, Dict[str, Any]]:
        system_prompt = None
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": 1024,
            "stream": stream,
        }
        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            anthropic_tools = []
            for tool in tools:
                if hasattr(tool, "fn") and hasattr(tool, "schema"):
                     anthropic_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.args_schema.model_json_schema()
                    })
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        return "/messages", payload

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        content_blocks = response_data.get("content", [])
        text_content = ""
        tool_calls = []

        if isinstance(content_blocks, list):
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    from ..types import ToolCall
                    tool_calls.append(ToolCall(
                        id=block.get("id"),
                        name=block.get("name"),
                        arguments=block.get("input")
                    ))
        
        usage_data = response_data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        return ModelResponse(
            text=text_content, 
            raw=response_data,
            usage=usage,
            provider="anthropic",
            tool_calls=tool_calls if tool_calls else None
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
