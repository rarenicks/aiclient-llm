import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage

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
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

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
