import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage, Text, Image, ToolMessage
from ..utils import encode_image

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
        
        formatted_messages = []
        current_tool_results = []
        
        for msg in messages:
            # Check if we need to flush tool results before processing a non-tool message
            if not isinstance(msg, ToolMessage) and current_tool_results:
                formatted_messages.append({
                    "role": "user",
                    "content": current_tool_results
                })
                current_tool_results = []

            if isinstance(msg, ToolMessage):
                current_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content
                })
                continue

            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
                 # Assistant with tool use
                 content_parts = []
                 # Add text content if exists
                 if msg.content:
                     content_parts.append({"type": "text", "text": msg.content})
                 
                 for tc in msg.tool_calls:
                     content_parts.append({
                         "type": "tool_use",
                         "id": tc.id,
                         "name": tc.name,
                         "input": tc.arguments
                     })
                 formatted_messages.append({"role": "assistant", "content": content_parts})
            else:
                if isinstance(msg.content, str):
                    formatted_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg.content, list):
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, str):
                            content_parts.append({"type": "text", "text": part})
                        elif isinstance(part, Text):
                             content_parts.append({"type": "text", "text": part.text})
                        elif isinstance(part, Image):
                            media_type, b64 = encode_image(part)
                            if not b64:
                                raise ValueError("Anthropic requires base64/path image.")
                            
                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64
                                }
                            })
                    formatted_messages.append({"role": msg.role, "content": content_parts})
        
        # Flush remaining tool results
        if current_tool_results:
             formatted_messages.append({
                "role": "user",
                "content": current_tool_results
             })

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

        return f"{self.base_url}/messages", payload

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
