import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data_types import (
    BaseMessage,
    Image,
    ModelResponse,
    StreamChunk,
    Text,
    ToolMessage,
    Usage,
)
from .base import Provider


class GoogleProvider(Provider):
    def __init__(self, api_key: str, base_url: str = None, api_version: str = "v1beta"):
        self.api_key = api_key
        # Default to v1beta unless base_url is provided
        if base_url:
            self._base_url = base_url.rstrip("/")
        else:
            self._base_url = f"https://generativelanguage.googleapis.com/{api_version}"
        self._buffer = ""

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def prepare_request(
        self,
        model: str,
        messages: List[BaseMessage],
        tools: List[Any] = None,
        stream: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        top_k: int = None,
        stop: Union[str, List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
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
                            parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": part.media_type,
                                        "data": b64,
                                    }
                                }
                            )

            if msg.role == "system":
                contents.append(
                    {"role": "user", "parts": [{"text": f"System: {msg.content}"}]}
                )
            elif isinstance(msg, ToolMessage):
                # Google expects: role="function",
                # parts=[{functionResponse: {name: ..., response: {result: ...}}}]
                # ... (already implemented)
                fname = msg.name or "unknown_tool"
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": fname,
                                    "response": {"name": fname, "content": msg.content},
                                }
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
                # Assistant with tool calls
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})

                for tc in msg.tool_calls:
                    parts.append(
                        {"functionCall": {"name": tc.name, "args": tc.arguments}}
                    )
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
                    funcs.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.args_schema.model_json_schema(),
                        }
                    )
            if funcs:
                payload["tools"] = [{"function_declarations": funcs}]

        if temperature is not None:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["temperature"] = temperature

        if max_tokens is not None:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        if top_p is not None:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["topP"] = top_p

        if top_k is not None:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["topK"] = top_k

        if stop is not None:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["stopSequences"] = (
                [stop] if isinstance(stop, str) else stop
            )

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
                    from ..data_types import ToolCall

                    tool_calls.append(
                        ToolCall(
                            id="call_" + fc["name"],  # No ID in Gemini 1.5 usually?
                            name=fc["name"],
                            arguments=fc["args"],
                        )
                    )
        except (KeyError, IndexError, TypeError):
            pass

        meta = response_data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=meta.get("promptTokenCount", 0),
            output_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
            cache_read_input_tokens=meta.get("cachedContentTokenCount", 0),
        )

        return ModelResponse(
            text=content,
            raw=response_data,
            usage=usage,
            provider="google",
            tool_calls=tool_calls if tool_calls else None,
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
        if not s:
            return None

        # Strip leading brackets/commas if they are just overhead
        if s.startswith("["):
            s = s[1:].strip()
        if s.startswith(","):
            s = s[1:].strip()

        # If it's just closing bracket, ignore
        if s == "]":
            self._buffer = ""
            return None

        # Heuristic: Only try parsing if it looks like we might find a closing brace
        # at end. But since we strip trailing commas for the trial, we need to be
        # careful.
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

            self._buffer = ""  # Clear buffer on success

            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return StreamChunk(text=text, delta=text)
            except (KeyError, IndexError):
                return None

        except json.JSONDecodeError:
            # Continue buffering
            return None

    def prepare_embeddings_request(
        self, model: str, input: Union[str, List[str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Prepare embeddings request for Google Gemini."""
        # Google uses batchEmbedContents for batch and embedContent for single
        # Model format: models/text-embedding-004
        if not model.startswith("models/"):
            model = f"models/{model}"

        if isinstance(input, str):
            # Single embedding request
            endpoint = f"{self.base_url}/{model}:embedContent?key={self.api_key}"
            data = {"model": model, "content": {"parts": [{"text": input}]}}
        else:
            # Batch embedding request
            endpoint = f"{self.base_url}/{model}:batchEmbedContents?key={self.api_key}"
            requests = []
            for text in input:
                requests.append(
                    {"model": model, "content": {"parts": [{"text": text}]}}
                )
            data = {"requests": requests}

        return endpoint, data

    def parse_embeddings_response(
        self, response_data: Dict[str, Any]
    ) -> Union[List[float], List[List[float]]]:
        """Parse Google Gemini embeddings response."""
        if "embedding" in response_data:
            # Single embedding response
            return response_data["embedding"]["values"]
        elif "embeddings" in response_data:
            # Batch embedding response
            return [emb["values"] for emb in response_data["embeddings"]]
        else:
            raise ValueError(f"Invalid embedding response: {response_data}")
