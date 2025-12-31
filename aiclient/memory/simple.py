from typing import List, Dict, Any, Optional
from ..data_types import BaseMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage
from .base import Memory

class ConversationMemory(Memory):
    """
    Simple memory that stores all messages in a list.
    """
    def __init__(self):
        self._messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def save(self) -> Dict[str, Any]:
        # Serialize messages
        return {
            "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in self._messages]
        }

    def load(self, data: Dict[str, Any]) -> None:
        raw_msgs = data.get("messages", [])
        self._messages = []
        # Rudimentary deserialization - ideal would be pydantic adapter
        for m in raw_msgs:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                self._messages.append(UserMessage(content=content))
            elif role == "model" or role == "assistant":
                self._messages.append(AssistantMessage(content=content, tool_calls=m.get("tool_calls")))
            elif role == "system":
                self._messages.append(SystemMessage(content=content))
            elif role == "tool":
                self._messages.append(ToolMessage(
                    tool_call_id=m.get("tool_call_id", "unknown"),
                    name=m.get("name", "unknown"),
                    content=str(content)
                ))

class SlidingWindowMemory(ConversationMemory):
    """
    Memory that keeps only the last N messages.
    Always preserves SystemMessages if present at start.
    """
    def __init__(self, max_messages: int = 10):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        self._truncate()

    def _truncate(self):
        if len(self._messages) <= self.max_messages:
            return
            
        # Identify System Messages to keep
        system_msgs = [m for m in self._messages if isinstance(m, SystemMessage)]
        
        # We want to keep (System Messages) + (Last K messages)
        # But total count must be <= max_messages
        
        remaining_slots = self.max_messages - len(system_msgs)
        if remaining_slots < 0:
            # Degenerate case: more system messages than max allowed. Keep max allowed.
            self._messages = self._messages[-self.max_messages:]
            return

        # Keep system messages + last remaining_slots
        others = [m for m in self._messages if not isinstance(m, SystemMessage)]
        keep_others = others[-remaining_slots:] if remaining_slots > 0 else []
        
        self._messages = system_msgs + keep_others
