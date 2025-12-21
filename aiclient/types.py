from typing import Any, Dict, Optional, Literal, List, Protocol, Union
from pydantic import BaseModel, Field

class Text(BaseModel):
    text: str

class Image(BaseModel):
    path: Optional[str] = None
    url: Optional[str] = None
    media_type: str = "image/jpeg"
    base64_data: Optional[str] = None

class BaseMessage(BaseModel):
    role: str
    content: Union[str, List[Union[str, Text, Image]]]

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
