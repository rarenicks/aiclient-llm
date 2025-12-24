# Conversation Memory 💭

Managing conversation history is critical for building chatbots and agents. `aiclient` provides a flexible memory system to handle this automatically.

## Usage

### Basic Memory

`ConversationMemory` stores all messages in a list.

```python
from aiclient import Client, Agent
from aiclient.memory import ConversationMemory

agent = Agent(
    model=Client().chat("gpt-4o"),
    memory=ConversationMemory()
)

# Run conversation
agent.run("Hi, my name is Alice.")
agent.run("What is my name?") # Agent will remember "Alice"
```

### Sliding Window Memory

Use `SlidingWindowMemory` to keep context size manageable by keeping only the most recent N messages. It **always preserves** the initial System Message if present.

```python
from aiclient.memory import SlidingWindowMemory

# Keep System Message + Last 10 interactions
memory = SlidingWindowMemory(max_messages=11)
```

## Persistence

You can save and load memory state (e.g., to a database or file).

```python
# Save
state = memory.save()
# state is a dict, easily JSON serializable

# Load
new_memory = ConversationMemory()
new_memory.load(state)
```
