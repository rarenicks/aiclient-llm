from typing import Any, Dict, Optional, Literal, List, Protocol
from pydantic import BaseModel

class BaseMessage(BaseModel):
    role: str
    content: str

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"

class UserMessage(BaseMessage):
    role: Literal["user"] = "user"

class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

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
