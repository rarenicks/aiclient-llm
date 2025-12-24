import pytest
import logging
from aiclient.observability import TracingMiddleware
from aiclient.data_types import UserMessage, ModelResponse, Usage

def test_tracing_middleware(caplog):
    caplog.set_level(logging.INFO)
    
    middleware = TracingMiddleware()
    
    # 1. Before Request
    messages = [UserMessage(content="Hello")]
    middleware.before_request("model", messages)
    
    # Check log
    assert "Trace[" in caplog.text
    assert "Request to model" in caplog.text
    
    # 2. After Response
    resp = ModelResponse(
        text="Hi", 
        raw={}, 
        usage=Usage(total_tokens=10),
        provider="test"
    )
    middleware.after_response(resp)
    
    assert "Response from test" in caplog.text
    assert "Tokens: 10" in caplog.text
