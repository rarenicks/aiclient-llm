# Agents & Model Context Protocol (MCP) 🤖

`aiclient` includes a built-in agent framework that simplifies building tool-using AI applications. It supports both standard Python functions as tools and the open standard **Model Context Protocol (MCP)** for connecting to external systems.

## The Agent Framework

The `Agent` class orchestrates a ReAct (Reason + Act) loop:
1. The model reasons about the user's request.
2. The model decides to call a tool.
3. The Agent executes the tool and feeds the result back.
4. The loop continues until the task is complete.

### Basic Usage with Python Functions

You can turn any Python function into a tool just by passing it to the agent. Type hints and docstrings are crucial—they are converted into the tool schema.

```python
from aiclient import Client, Agent

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real app, call a weather API here
    return f"Sunny, 25°C in {location}"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}: ..."

client = Client()
agent = Agent(
    model=client.chat("gpt-4o"),
    tools=[get_weather, search_web],
    max_steps=10  # Prevent infinite loops
)

result = agent.run("What's the weather in Tokyo and find me top restaurants there?")
print(result)
```

## Model Context Protocol (MCP) 🔌

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting AI models to external data and tools. `aiclient` has native support for MCP clients.

This means you can connect your agent to **any** MCP-compliant server (e.g., GitHub, PostgreSQL, Google Drive, filesystem) without writing any tool adapters yourself.

### Connecting to MCP Servers

You configure MCP servers by specifying how to launch them (stdio connection).

```python
agent = Agent(
    model=client.chat("gpt-4o"),
    mcp_servers={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        }
    }
)

# The agent now automagically has access to tools like:
# - filesystem_read_file
# - filesystem_list_directory
# - github_create_issue
# ... and more!

async with agent:
    await agent.run_async("Check the project README and create a GitHub issue for any TODOs found.")
```

### Docker Support

You can also run MCP servers via Docker for security and isolation:

```python
mcp_servers = {
    "fetch": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "mcp/fetch"] # Official MCP fetch tool
    }
}
```

## Advanced Agent Configuration

### Memory Management

Agents maintain conversation history. You can customize the memory implementation:

```python
from aiclient.memory import SlidingWindowMemory

# Keep only the last 20 messages to save context window
memory = SlidingWindowMemory(max_messages=20)

agent = Agent(
    model=client.chat("gpt-4o"),
    memory=memory
)
```

### System Prompts

Customize the agent's persona and instructions:

```python
agent = Agent(
    model=client.chat("gpt-4o"),
    system_prompt="You are a helpful coding assistant. Always explain your code."
)
```
