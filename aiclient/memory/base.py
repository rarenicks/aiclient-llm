from typing import List, Protocol, Dict, Any, Optional
from ..types import BaseMessage

class Memory(Protocol):
    """
    Protocol for conversation memory systems.
    """
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to memory."""
        ...

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve stored messages."""
        ...

    def clear(self) -> None:
        """Clear all messages."""
        ...

    def save(self) -> Dict[str, Any]:
        """Export state to a dictionary."""
        ...

    def load(self, data: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        ...
