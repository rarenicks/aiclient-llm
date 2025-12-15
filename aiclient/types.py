from typing import Any, Dict, Optional
from pydantic import BaseModel

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class ModelResponse(BaseModel):
    """Standardized response from any AI provider."""
    text: str
    raw: Dict[str, Any]
    usage: Usage = Usage()
    provider: str = "unknown"

class StreamChunk(BaseModel):
    """Standardized stream chunk."""
    text: str
    delta: str
