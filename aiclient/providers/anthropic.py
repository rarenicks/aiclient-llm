import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from .base import Provider
from ..data_types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage, Text, Image, ToolMessage
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

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False, response_schema: Optional[Dict[str, Any]] = None, strict: bool = False, temperature: float = None) -> Tuple[str, Dict[str, Any]]:
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
                if msg.cache_control:
                    system_prompt = [{
                        "type": "text", 
                        "text": msg.content, 
                        "cache_control": {"type": msg.cache_control}
                    }]
                else:
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
                    content_block = {"type": "text", "text": msg.content}
                    if msg.cache_control:
                        content_block["cache_control"] = {"type": msg.cache_control}
                    formatted_messages.append({"role": msg.role, "content": [content_block]})
                elif isinstance(msg.content, list):
                    content_parts = []
                    for i, part in enumerate(msg.content):
                        if isinstance(part, str):
                            block = {"type": "text", "text": part}
                        elif isinstance(part, Text):
                             block = {"type": "text", "text": part.text}
                        elif isinstance(part, Image):
                            # Anthropic prefers base64 even for URLs usually, or check docs. 
                            # For consistency with v0.2 logic, we enforce base64.
                            # If it has a URL, we fetch it via to_base64() if we want consistency,
                            # OR we just support base64/path here. 
                            # The Image.to_base64() helper handles path/url/base64 unified.
                            
                            b64 = part.to_base64()
                            media_type = part.media_type
                            
                            block = {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64
                                }
                            }
                        
                        # Apply cache_control to the LAST block if set on message level?
                        # Or should we support per-block caching in types? 
                        # For now, let's apply to the LAST block if msg.cache_control is set, 
                        # as Anthropic usually caches up to a point.
                        # However, for fine-grained control, we might need it on Text/Image types later.
                        # V0.4 MVP: Apply to last block of the message.
                        if msg.cache_control and i == len(msg.content) - 1:
                            block["cache_control"] = {"type": msg.cache_control}

                        content_parts.append(block)
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
        
        if temperature is not None:
            payload["temperature"] = temperature

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
                    from ..data_types import ToolCall
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
            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0)
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

    def prepare_embeddings_request(self, model: str, input: Union[str, List[str]]) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError("Anthropic does not expose a public embeddings API via this client yet.")
    
    def parse_embeddings_response(self, response_data: Dict[str, Any]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("Anthropic does not expose a public embeddings API via this client yet.")

