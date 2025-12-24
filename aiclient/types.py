from typing import Any, Dict, Optional, Literal, List, Protocol, Union
from pydantic import BaseModel, Field

class Text(BaseModel):
    text: str

import base64
import httpx
from pathlib import Path

class Image(BaseModel):
    path: Optional[str] = None
    url: Optional[str] = None
    media_type: str = "image/jpeg"
    base64_data: Optional[str] = None

    def to_base64(self) -> str:
        """
        Returns the base64 encoded string of the image.
        Resolves path or url if base64_data is not already set.
        """
        if self.base64_data:
            return self.base64_data
            
        if self.path:
            p = Path(self.path)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {self.path}")
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
                
        if self.url:
            # Synchronous fetch for simplicity in this helper, 
            # or we assume the user provides base64 for perf.
            # Using httpx for convenience.
            resp = httpx.get(self.url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
            
        raise ValueError("Image must have path, url, or base64_data")

class BaseMessage(BaseModel):
    role: str
    content: Union[str, List[Union[str, Text, Image]]]
    cache_control: Optional[Literal["ephemeral"]] = None

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"

class UserMessage(BaseMessage):
    role: Literal["user"] = "user"

class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: Optional[str] = None
    content: str

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    tool_calls: Optional[List[ToolCall]] = None

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = 0
    cache_read_input_tokens: Optional[int] = 0

class ModelResponse(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def usage(self) -> Optional[Usage]: ...
    @property
    def tool_calls(self) -> Optional[List[ToolCall]]: ...

class StreamChunk(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def delta(self) -> str: ... # Alias for text to match user preference

class ModelResponse(BaseModel):
    """Standardized response from any AI provider."""
    text: str
    raw: Dict[str, Any]
    usage: Optional[Usage] = None
    provider: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class StreamChunk(BaseModel):
    """Standardized stream chunk."""
    text: str
    delta: str
