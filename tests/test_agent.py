"""
Tests for Agent functionality with tool use.
"""
import pytest
from unittest.mock import MagicMock
from aiclient.agent import Agent
from aiclient.models.chat import ChatModel
from aiclient.data_types import ModelResponse, ToolCall, Usage
from aiclient.tools.base import Tool


def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"


def get_time(timezone: str = "UTC") -> str:
    """Get current time."""
    return f"12:00 PM {timezone}"


def test_agent_single_tool_call():
    """Test agent with single tool execution."""
    # Mock model
    mock_model = MagicMock(spec=ChatModel)

    # First response: Model calls tool
    mock_model.generate_async.side_effect = [
        ModelResponse(
            text="I'll check the weather",
            raw={},
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"location": "Tokyo"})
            ]
        ),
        # Second response: Final answer
        ModelResponse(
            text="The weather in Tokyo is sunny!",
            raw={},
            tool_calls=None
        )
    ]

    agent = Agent(model=mock_model, tools=[get_weather], max_steps=5)
    result = agent.run("What's the weather in Tokyo?")

    assert result == "The weather in Tokyo is sunny!"
    assert mock_model.generate_async.call_count == 2


def test_agent_multiple_tools():
    """Test agent with multiple tools available."""
    mock_model = MagicMock(spec=ChatModel)

    # Model calls two tools in sequence
    mock_model.generate_async.side_effect = [
        # Call weather tool
        ModelResponse(
            text="Checking weather",
            raw={},
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"location": "Paris"})]
        ),
        # Call time tool
        ModelResponse(
            text="Checking time",
            raw={},
            tool_calls=[ToolCall(id="call_2", name="get_time", arguments={"timezone": "CET"})]
        ),
        # Final answer
        ModelResponse(
            text="In Paris it's sunny and 12:00 PM CET",
            raw={},
            tool_calls=None
        )
    ]

    agent = Agent(model=mock_model, tools=[get_weather, get_time], max_steps=10)
    result = agent.run("What's the weather and time in Paris?")

    assert "sunny" in result.lower() or "paris" in result.lower()
    assert mock_model.generate_async.call_count == 3


def test_agent_max_steps_reached():
    """Test agent stops after max_steps."""
    mock_model = MagicMock(spec=ChatModel)

    # Always return tool calls (infinite loop)
    mock_model.generate_async.return_value = ModelResponse(
        text="Calling tool",
        raw={},
        tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"location": "Test"})]
    )

    agent = Agent(model=mock_model, tools=[get_weather], max_steps=3)
    result = agent.run("Test")

    assert result == "Max steps reached"
    assert mock_model.generate_async.call_count == 3


def test_agent_tool_error_handling():
    """Test agent handles tool execution errors."""
    mock_model = MagicMock(spec=ChatModel)

    def failing_tool(param: str) -> str:
        """A tool that always fails."""
        raise ValueError("Tool failed!")

    # Model calls failing tool, then gives answer
    mock_model.generate_async.side_effect = [
        ModelResponse(
            text="Calling tool",
            raw={},
            tool_calls=[ToolCall(id="call_1", name="failing_tool", arguments={"param": "test"})]
        ),
        ModelResponse(
            text="I encountered an error",
            raw={},
            tool_calls=None
        )
    ]

    agent = Agent(model=mock_model, tools=[Tool.from_fn(failing_tool)], max_steps=5)
    result = agent.run("Test")

    # Should handle error gracefully
    assert result
    assert mock_model.generate_async.call_count == 2


def test_agent_no_tool_calls():
    """Test agent when model doesn't need tools."""
    mock_model = MagicMock(spec=ChatModel)

    # Direct answer without tools
    mock_model.generate_async.return_value = ModelResponse(
        text="The answer is 42",
        raw={},
        tool_calls=None
    )

    agent = Agent(model=mock_model, tools=[get_weather], max_steps=5)
    result = agent.run("What is the meaning of life?")

    assert result == "The answer is 42"
    assert mock_model.generate_async.call_count == 1


# Requires pytest-asyncio
# @pytest.mark.asyncio
# async def test_agent_async_tool():
#     """Test agent with async tools."""
#     mock_model = MagicMock(spec=ChatModel)
#
#     async def async_weather(location: str) -> str:
#         """Async weather tool."""
#         return f"Async weather in {location}"
#
#     mock_model.generate_async.side_effect = [
#         ModelResponse(
#             text="Checking weather",
#             raw={},
#             tool_calls=[ToolCall(id="call_1", name="async_weather", arguments={"location": "NYC"})]
#         ),
#         ModelResponse(
#             text="Weather checked!",
#             raw={},
#             tool_calls=None
#         )
#     ]
#
#     agent = Agent(model=mock_model, tools=[async_weather], max_steps=5)
#     result = await agent.run_async("Weather in NYC?")
#
#     assert result == "Weather checked!"
#     assert mock_model.generate_async.call_count == 2
