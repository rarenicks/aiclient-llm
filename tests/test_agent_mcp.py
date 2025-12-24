import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from aiclient import Agent, Client
from aiclient.data_types import ModelResponse, ToolCall
from aiclient.mcp.manager import MCPServerManager

@pytest.mark.asyncio
async def test_agent_mcp_lifecycle():
    # Mock Manager
    with patch("aiclient.agent.MCPServerManager") as mock_manager_cls:
        mock_manager = AsyncMock()
        # add_server is sync in reality, so mock it as sync
        mock_manager.add_server = MagicMock()
        
        # Setup context manager returns
        mock_manager.__aenter__.return_value = mock_manager
        mock_manager.list_global_tools.return_value = []
        
        mock_manager_cls.return_value = mock_manager
        
        mock_model = AsyncMock()
        mock_model.generate_async.return_value = ModelResponse(text="Done", raw={}, provider="test")
        
        agent = Agent(model=mock_model, mcp_servers={"test": {"command": "echo"}})
        
        async with agent:
            # Manager should be started
            mock_manager.__aenter__.assert_awaited()
            
            await agent.run_async("hi")
            
            # Tools should be fetched
            mock_manager.list_global_tools.assert_awaited()
            
        # Manager should be stopped
        mock_manager.__aexit__.assert_awaited()

@pytest.mark.asyncio
async def test_agent_call_mcp_tool():
    # Verify tool execution flow
    with patch("aiclient.agent.MCPServerManager") as mock_manager_cls:
        mock_manager = AsyncMock()
        mock_manager.add_server = MagicMock()
        mock_manager.__aenter__.return_value = mock_manager
        
        # Mock available tool
        mock_tool_def = MagicMock()
        mock_tool_def.name = "get_weather"
        mock_tool_def.description = "Get weather"
        mock_manager.list_global_tools.return_value = [mock_tool_def]
        
        # Mock execution result
        mock_manager.call_tool.return_value = "Sunny"
        
        mock_manager_cls.return_value = mock_manager
        
        # Mock Model to request tool call then finish
        mock_model = AsyncMock()
        
        # 1st response: Tool Call
        resp1 = ModelResponse(
             text="Calling tool", 
             raw={}, 
             provider="test",
             tool_calls=[ToolCall(id="1", name="get_weather", arguments={"city": "SF"})]
        )
        # 2nd response: Final
        resp2 = ModelResponse(text="It is Sunny", raw={}, provider="test")
        
        mock_model.generate_async.side_effect = [resp1, resp2]
        
        agent = Agent(model=mock_model, mcp_servers={"test": {"command": "echo"}})
        
        async with agent:
            result = await agent.run_async("Weather?")
            
            # Verify call_tool was invoked on manager
            mock_manager.call_tool.assert_awaited_with("get_weather", {"city": "SF"})
            
            # Verify result passed to model
            # Note: Checking history is harder with mocks, but model should be called twice
            assert mock_model.generate_async.await_count == 2
            assert result == "It is Sunny"
@pytest.mark.asyncio
async def test_agent_multiple_mcp_tools():
    # Verify strict routing with multiple tools (catch closure capture bug)
    with patch("aiclient.agent.MCPServerManager") as mock_manager_cls:
        mock_manager = AsyncMock()
        mock_manager.add_server = MagicMock()
        mock_manager.__aenter__.return_value = mock_manager
        
        # Mock 2 available tools
        t1 = MagicMock(); t1.name = "tool_a"; t1.description = "A"
        t2 = MagicMock(); t2.name = "tool_b"; t2.description = "B"
        mock_manager.list_global_tools.return_value = [t1, t2]
        
        mock_manager.call_tool.return_value = "Result"
        mock_manager_cls.return_value = mock_manager
        
        # Mock Model to call FIRST tool (tool_a)
        mock_model = AsyncMock()
        # 1. Tool Call for tool_a
        resp1 = ModelResponse(
             text="Calling A", 
             raw={}, 
             provider="test",
             tool_calls=[ToolCall(id="1", name="tool_a", arguments={})]
        )
        # 2. Final
        resp2 = ModelResponse(text="Done", raw={}, provider="test")
        mock_model.generate_async.side_effect = [resp1, resp2]
        
        agent = Agent(model=mock_model, mcp_servers={"test": {"command": "echo"}})
        
        async with agent:
            await agent.run_async("use A")
            
            # CRITICAL: Verify tool_a was called, NOT tool_b
            # If closure bug exists, it might call tool_b (last in loop) instead of tool_a
            mock_manager.call_tool.assert_awaited_with("tool_a", {})
