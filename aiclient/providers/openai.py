import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from .base import Provider
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage, Text, Image, ToolMessage
from ..utils import encode_image

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

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False) -> Tuple[str, Dict[str, Any]]:
        url = "https://api.openai.com/v1/chat/completions"
        if model.startswith("grok"):
             url = "https://api.x.ai/v1/chat/completions"

        formatted_messages = []
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
                 tcs = []
                 for tc in msg.tool_calls:
                     tcs.append({
                         "id": tc.id,
                         "type": "function",
                         "function": {
                             "name": tc.name,
                             "arguments": json.dumps(tc.arguments)
                         }
                     })
                 formatted_messages.append({
                     "role": "assistant",
                     "content": msg.content, 
                     "tool_calls": tcs
                 })
            elif isinstance(msg.content, str):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                content_parts = []
                for part in msg.content:
                    if isinstance(part, str):
                        content_parts.append({"type": "text", "text": part})
                    elif isinstance(part, Text):
                         content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, Image):
                        if part.url:
                            image_url_val = part.url
                        else:
                            media_type, b64 = encode_image(part)
                            image_url_val = f"data:{media_type};base64,{b64}"
                            image_url_val = f"data:{media_type};base64,{b64}"
                        
                        img_payload = {"url": image_url_val}
                        # xAI does not support 'detail' param apparently? Or strictly follows standard?
                        # Let's keep detail for non-grok or default.
                        if not model.startswith("grok"):
                             img_payload["detail"] = "auto"

                        content_parts.append({
                            "type": "image_url",
                            "image_url": img_payload
                        })
                formatted_messages.append({"role": msg.role, "content": content_parts})
            else:
                 # Fallback for simple message
                 formatted_messages.append({"role": msg.role, "content": str(msg.content)})

        data = {
            "model": model,
            "messages": formatted_messages,
            "stream": stream,
        }
        
        # Tool serialization
        if tools:
            openai_tools = []
            for tool in tools:
                # Assuming tool is a Tool object from aiclient.tools
                # We need to map it to OpenAI's expected format
                if hasattr(tool, "fn") and hasattr(tool, "schema"):
                     openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.fn.__doc__ or "",
                            "parameters": tool.args_schema.model_json_schema()
                        }
                    })
            if openai_tools:
                data["tools"] = openai_tools

        return url, data

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        message = response_data["choices"][0]["message"]
        content = message.get("content") or ""
        
        # Extract tools
        tool_calls = []
        if message.get("tool_calls"):
            raw_calls = message["tool_calls"]
            from ..types import ToolCall
            for rc in raw_calls:
                tool_calls.append(ToolCall(
                    id=rc["id"],
                    name=rc["function"]["name"],
                    arguments=json.loads(rc["function"]["arguments"])
                ))

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
            provider="openai",
            tool_calls=tool_calls if tool_calls else None
        )

    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        raw_obj = chunk.get("raw")
        if isinstance(raw_obj, bytes):
            raw_str = raw_obj.decode("utf-8")
        else:
            raw_str = str(raw_obj)
        
        # print(f"DEBUG CHUNK: {raw_str!r}") # Uncomment for debug


        if not raw_str.strip():
            return None
            
        if not raw_str.startswith("data: "):
            return None
        
        data_str = raw_str[6:].strip() # Remove "data: "
        if data_str == "[DONE]":
            return None
            
        try:
            import json
            data = json.loads(data_str)
            delta = data["choices"][0]["delta"].get("content", "")
            if not delta: # Might be empty or tool call
                return None
            return StreamChunk(text=delta, delta=delta)
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
