import asyncio
from typing import List, Any, Optional, Dict
from .models.chat import ChatModel
from .data_types import UserMessage, ToolMessage, AssistantMessage, BaseMessage
from .tools.base import Tool
from .mcp import MCPServerManager
from .memory import Memory, ConversationMemory

class Agent:
    """
    An agent that can use tools (local and MCP) to solve tasks.
    """
    def __init__(
        self, 
        model: ChatModel, 
        tools: List[Any] = [], 
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        max_steps: int = 10,
        memory: Optional[Memory] = None
    ):
        self.model = model
        self.max_steps = max_steps
        self.memory = memory or ConversationMemory()
        
        # Local Tools
        self.tools = []
        self._tool_map: Dict[str, Tool] = {}
        for t in tools:
            if isinstance(t, Tool):
                tool_instance = t
            else:
                tool_instance = Tool.from_fn(t)
            self.tools.append(tool_instance)
            self._tool_map[tool_instance.name] = tool_instance
            
        # MCP Manager
        self.mcp_manager = MCPServerManager()
        if mcp_servers:
            for name, config in mcp_servers.items():
                self.mcp_manager.add_server(
                    name=name,
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env")
                )
            
        # self.history is now managed by self.memory if desired, or we just sync them?
        # Ideally, we rely on memory.get_messages().
        # But for backward compat/simplicity in this refactor, we can ensure memory is updated.

    async def __aenter__(self):
        await self.mcp_manager.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mcp_manager.__aexit__(exc_type, exc_val, exc_tb)

    async def run_async(self, prompt: str) -> str:
        """Asynchronous run loop."""
        # Add user prompt to memory
        self.memory.add_message(UserMessage(content=prompt))
        
        # 1. Fetch MCP tools (if servers are active)
        if self.mcp_manager.has_servers and not self.mcp_manager.is_active:
            print("WARNING: MCP servers configured but Agent not running in 'async with' context. MCP tools will be unavailable.")
            # Simple run (no tools)
            history = self.memory.get_messages()
            response = await self.model.generate_async(history, tools=self.tools)
            self.memory.add_message(AssistantMessage(content=response.text))
            return response.text
        
        mcp_tools_schemas = await self.mcp_manager.list_global_tools()
        
        # Convert MCP schemas to our Tool objects
        for tool_def in mcp_tools_schemas:
            name = tool_def.name
            
            # Create a localized runner for this tool
            async def mcp_runner_wrapper(_name=name, **kwargs):
                return await self.mcp_manager.call_tool(_name, kwargs)
            
            mcp_runner_wrapper.__name__ = name

            tool_obj = Tool(
                name=name,
                fn=mcp_runner_wrapper,
                schema=None, # No Pydantic schema
                description=tool_def.description or "",
                raw_schema=tool_def.inputSchema
            )
            
            if name not in self._tool_map:
                self.tools.append(tool_obj)
                self._tool_map[name] = tool_obj
        
        all_tools = self.tools
        
        for _ in range(self.max_steps):
            history = self.memory.get_messages()
            response = await self.model.generate_async(history, tools=all_tools)
            
            assistant_msg = AssistantMessage(
                content=response.text,
                tool_calls=response.tool_calls
            )
            self.memory.add_message(assistant_msg)
            
            if not response.tool_calls:
                return response.text
                
            for tc in response.tool_calls:
                tool = self._tool_map.get(tc.name)
                result = None
                
                if tool:
                    try:
                        if asyncio.iscoroutinefunction(tool.fn):
                            result = await tool.fn(**tc.arguments)
                        else:
                            result = tool.fn(**tc.arguments)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    try:
                        result = await self.mcp_manager.call_tool(tc.name, tc.arguments)
                    except Exception as e:
                         result = f"Error: Tool {tc.name} not found or failed: {e}"

                self.memory.add_message(ToolMessage(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=str(result)
                ))
                    
        return "Max steps reached"

    def run(self, prompt: str) -> str:
        """Synchronous run loop wrapper."""
        return asyncio.run(self.run_async(prompt))
