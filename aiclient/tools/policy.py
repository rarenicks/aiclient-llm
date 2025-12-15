from .base import Tool
from pydantic import BaseModel

class PolicyCheckSchema(BaseModel):
    text: str

def check_policy(text: str) -> bool:
    """Check if text complies with policy."""
    return "forbidden" not in text

policy_tool = Tool(
    name="check_policy",
    func=check_policy,
    args_schema=PolicyCheckSchema,
    description="Checks if the text complies with content policy."
)
