
import pytest
from unittest.mock import MagicMock
from aiclient.middleware import Middleware
from aiclient.data_types import ToolMessage, ToolCall, ModelResponse, Usage
from aiclient.models.chat import ChatModel

class ToolTrimmerMiddleware(Middleware):
    def on_tool_result(self, tool_result: ToolMessage) -> ToolMessage:
        if len(tool_result.content) > 10:
            tool_result.content = tool_result.content[:10] + "..."
        return tool_result
    
    def on_tool_call(self, tool_call: ToolCall) -> ToolCall:
        tool_call.arguments["trimmed"] = True
        return tool_call

    def before_request(self, model, prompt):
        return prompt

    def after_response(self, response):
        return response

def test_middleware_on_tool_result():
    """Test that tool results are trimmed by middleware."""
    middleware = ToolTrimmerMiddleware()
    
    # Mock provider and transport
    provider = MagicMock()
    transport = MagicMock()
    
    # Setup mock response
    provider.prepare_request.return_value = ("endpoint", {})
    provider.parse_response.return_value = ModelResponse(
        text="Response",
        raw={},
        usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
        provider="mock"
    )
    
    model = ChatModel("mock-model", provider, transport, middlewares=[middleware])
    
    # Create a long tool message
    long_content = "This is a very long tool output that should be trimmed"
    tool_msg = ToolMessage(tool_call_id="call_1", content=long_content)
    
    model.generate([tool_msg])
    
    # Check that prepare_request received the trimmed message
    call_args = provider.prepare_request.call_args
    messages_arg = call_args[0][1] # (model, messages, ...)
    assert len(messages_arg) == 1
    assert messages_arg[0].content == "This is a ..." # 10 chars + ...

def test_middleware_on_tool_call():
    """Test that tool calls are modified by middleware."""
    middleware = ToolTrimmerMiddleware()
    
    provider = MagicMock()
    transport = MagicMock()
    
    provider.prepare_request.return_value = ("endpoint", {})
    # Mock response with a tool call
    original_tool_call = ToolCall(id="call_1", name="search", arguments={"query": "foo"})
    provider.parse_response.return_value = ModelResponse(
        text="",
        raw={},
        usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
        provider="mock",
        tool_calls=[original_tool_call]
    )
    
    model = ChatModel("mock-model", provider, transport, middlewares=[middleware])
    
    response = model.generate("Hello")
    
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    # Check modification
    assert response.tool_calls[0].arguments["trimmed"] is True
