import contextlib
import asyncio
from typing import Dict, Any, List, Optional
from .client import MCPClient

class MCPServerManager:
    """
    Manages the lifecycle of MCP servers.
    """
    def __init__(self):
        self._tool_server_map: Dict[str, str] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._exit_stack = contextlib.AsyncExitStack()
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active
        
    @property
    def has_servers(self) -> bool:
        return bool(self._clients)

    def add_server(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Register a server config. Does not connect yet.
        """
        self._clients[name] = MCPClient(command, args, env)

    async def __aenter__(self):
        """
        Start all registered servers.
        """
        for name, client in self._clients.items():
            try:
                await self._exit_stack.enter_async_context(client)
            except Exception as e:
                # TODO: Log error, maybe skip this server or fail hard?
                print(f"Failed to start MCP server {name}: {e}")
                # For now, let's re-raise to be safe, or continue?
                # Re-raising is safer for consistency.
                raise e
        self._is_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        await self._exit_stack.aclose()


    async def get_client(self, name: str) -> MCPClient:
        return self._clients[name]
        
    async def list_global_tools(self) -> List[Dict[str, Any]]:
        """
        Aggregate tools from all connected servers.
        """
        all_tools = []
        # Clear map to rebuild
        self._tool_server_map.clear()
        
        for name, client in self._clients.items():
            try:
                tools = await client.list_tools()
                for t in tools:
                    all_tools.append(t)
                    self._tool_server_map[t.name] = name
            except Exception as e:
                # Log error
                pass
        return all_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        server_name = self._tool_server_map.get(name)
        if server_name:
            client = self._clients.get(server_name)
            if client:
                return await client.call_tool(name, arguments)
        
        # Fallback: Inefficient search (useful if map stale or not inited)
        # Or we can just raise error if we enforce list_tools called first.
        # Let's keep fallback for robustness in v0.3 dev.
        for client in self._clients.values():
            try:
                tools = await client.list_tools() 
                for tool in tools:
                    if tool.name == name:
                        # Found it! Update map for next time?
                        # We don't know the server name inside this loop easily unless iterating items.
                        return await client.call_tool(name, arguments)
            except:
                continue
        raise ValueError(f"Tool {name} not found on any server")
