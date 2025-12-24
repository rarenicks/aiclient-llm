import json
from typing import Any, Dict, Tuple, Optional, Union, List
from .base import Provider
from .base import Provider
from ..types import ModelResponse, StreamChunk, Usage, BaseMessage, UserMessage, Text, Image, ToolMessage
from ..utils import encode_image

class GoogleProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._buffer = ""

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def prepare_request(self, model: str, messages: List[BaseMessage], tools: List[Any] = None, stream: bool = False, response_schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Tuple[str, Dict[str, Any]]:
        self._buffer = ""
        contents = []
        contents = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, str):
                        parts.append({"text": part})
                    elif isinstance(part, Text):
                        parts.append({"text": part.text})
                    elif isinstance(part, Image):
                        # Gemini supports inlineData for base64. 
                        # We use to_base64() to handle URL/Path/Base64 unified.
                        b64 = part.to_base64()
                        if b64:
                             parts.append({
                                 "inlineData": {
                                     "mimeType": part.media_type,
                                     "data": b64
                                 }
                             })

            if msg.role == "system":
                 contents.append({"role": "user", "parts": [{"text": f"System: {msg.content}"}]})
            elif isinstance(msg, ToolMessage):
                 # Google expects: role="function", parts=[{functionResponse: {name: ..., response: {result: ...}}}]
                 # ... (already implemented)
                 fname = msg.name or "unknown_tool"
                 contents.append({
                     "role": "function", 
                     "parts": [{
                         "functionResponse": {
                             "name": fname,
                             "response": {"name": fname, "content": msg.content} 
                         }
                     }]
                 })
            elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
                 # Assistant with tool calls
                 parts = []
                 if msg.content:
                     parts.append({"text": msg.content})
                 
                 for tc in msg.tool_calls:
                     parts.append({
                         "functionCall": {
                             "name": tc.name,
                             "args": tc.arguments
                         }
                     })
                 contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": role, "parts": parts})

        # Endpoint selection
        method = "streamGenerateContent" if stream else "generateContent"
        endpoint = f"{self.base_url}/models/{model}:{method}?key={self.api_key}"
        
        payload = {"contents": contents}

        # Tool serialization
        if tools:
            funcs = []
            for tool in tools:
                 if hasattr(tool, "args_schema"):
                     funcs.append({
                         "name": tool.name,
                         "description": tool.description,
                         "parameters": tool.args_schema.model_json_schema()
                     })
            if funcs:
                payload["tools"] = [{"function_declarations": funcs}]

        return endpoint, payload

    def parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        content = ""
        tool_calls = []
        
        try:
            candidate = response_data["candidates"][0]
            parts = candidate["content"]["parts"]
            for part in parts:
                if "text" in part:
                    content += part["text"]
                if "functionCall" in part:
                    fc = part["functionCall"]
                    from ..types import ToolCall
                    tool_calls.append(ToolCall(
                        id="call_" + fc["name"], # No ID in Gemini 1.5 usually?
                        name=fc["name"],
                        arguments=fc["args"]
                    ))
        except (KeyError, IndexError, TypeError):
             pass
             
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
            provider="google",
            tool_calls=tool_calls if tool_calls else None
        )

    def parse_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamChunk]:
        raw_obj = chunk.get("raw")
        if isinstance(raw_obj, bytes):
            raw_str = raw_obj.decode("utf-8")
        else:
            raw_str = str(raw_obj)
        
        self._buffer += raw_str
        
        # Try to parse buffer
        s = self._buffer.strip()
        if not s: return None
        
        # Strip leading brackets/commas if they are just overhead
        if s.startswith("["): s = s[1:].strip()
        if s.startswith(","): s = s[1:].strip()
        
        # If it's just closing bracket, ignore
        if s == "]": 
            self._buffer = ""
            return None

        # Heuristic: Only try parsing if it looks like we might find a closing brace at end
        # But since we strip trailing commas for the trial, we need to be careful.
        # Let's clean up the buffer for parsing trial
        candidate_json = s
        if candidate_json.endswith(","):
            candidate_json = candidate_json[:-1]
        
        try:
            data = json.loads(candidate_json)
            # If successful, we have a complete object. 
            # RESET buffer for next object. 
            # WARNING: This assumes one object per chunk sequence. 
            # If buffer contained multiple objects, we lose the rest.
            # But line-by-line accumulation usually hits one object boundary at a time?
            # Actually, if we accumulate lines, we might get `} , {`.
            # If json.loads succeeds, it parsed ONE object? 
            # standard json.loads parses the whole string. 
            # If we have extra data, it fails.
            
            self._buffer = "" # Clear buffer on success
            
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return StreamChunk(text=text, delta=text)
            except (KeyError, IndexError):
                return None
                
        except json.JSONDecodeError:
            # Continue buffering
            return None 
