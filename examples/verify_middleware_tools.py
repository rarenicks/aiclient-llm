
import asyncio
from unittest.mock import MagicMock, AsyncMock
from aiclient.client import Client
from aiclient.middleware import Middleware
from aiclient.data_types import ToolCall, ToolMessage, ModelResponse, Usage

# 1. Define Middleware
class SecurityAuditMiddleware(Middleware):
    def __init__(self):
        self.triggered_call = False
        self.triggered_result = False

    def on_tool_call(self, tool_call: ToolCall) -> ToolCall:
        print(f"[Middleware] Intercepted Tool Call: {tool_call.name}({tool_call.arguments})")
        # Modify argument for auditing
        tool_call.arguments["audited"] = True
        self.triggered_call = True
        return tool_call

    def on_tool_result(self, tool_result: ToolMessage) -> ToolMessage:
        print(f"[Middleware] Intercepted Tool Result for {tool_result.tool_call_id}")
        # Trim result if too long
        if len(tool_result.content) > 20:
            print("[Middleware] Trimming tool result...")
            tool_result.content = tool_result.content[:20] + "...(trimmed)"
        self.triggered_result = True
        return tool_result
        
    # Passthrough required for other hooks
    def before_request(self, model, prompt): return prompt
    def after_response(self, response): return response

async def run_verification():
    print("--- Starting E2E Verification for Middleware Tools ---")
    
    # 2. Setup Client with Middleware
    audit_mw = SecurityAuditMiddleware()
    client = Client(openai_api_key="mock-key")
    client.add_middleware(audit_mw)
    
    # 3. Mock Provider to avoid real API calls
    # We need to reach into the client to mock the provider creation or the transport
    # Easiest is to mock the `chat` method's `transport.send_async`
    
    model = client.chat("gpt-4o")
    
    # Mocking the transport send to return a Tool Call
    mock_response_payload = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "safe_search",
                        "arguments": "{\"query\": \"python middleware\"}"
                    }
                }]
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    model.transport.send_async = AsyncMock(return_value=mock_response_payload)
    
    print("\n1. Testing on_tool_call...")
    response = await model.generate_async("Search for python middleware")
    
    if audit_mw.triggered_call:
        print("✅ Middleware triggered on_tool_call")
    else:
        print("❌ Middleware FAILED to trigger on_tool_call")
        
    tool_call = response.tool_calls[0]
    if tool_call.arguments.get("audited"):
        print(f"✅ Tool call arguments modified: {tool_call.arguments}")
    else:
        print(f"❌ Tool call arguments NOT modified: {tool_call.arguments}")

    print("\n2. Testing on_tool_result...")
    # Simulate feeding back a tool result
    tool_msg = ToolMessage(
        tool_call_id="call_123", 
        content="This is a very long sensitive result that should be trimmed by the middleware."
    )
    
    # For this test, we mock the response to be simple text
    mock_response_text = {
        "choices": [{"message": {"role": "assistant", "content": "I processed your result."}}],
        "usage": {}
    }
    model.transport.send_async = AsyncMock(return_value=mock_response_text)
    
    # Initialize middleware state
    audit_mw.triggered_result = False
    
    await model.generate_async([tool_msg])
    
    if audit_mw.triggered_result:
        print("✅ Middleware triggered on_tool_result")
    else:
        print("❌ Middleware FAILED to trigger on_tool_result")
        
    # Check if the message passed to transport was trimmed
    # internal transport mock call args
    call_args = model.transport.send_async.call_args
    # call_args[0] is (endpoint, data)
    # data["messages"] contains the messages sent
    sent_messages = call_args[0][1]["messages"] # list of dicts or objects depending on prepare_request logic
    
    # The provider `prepare_request` converts objects to dicts.
    # We need to verify what `prepare_request` received OR what `send_async` received.
    # In `ChatModel.generate_async`, `messages` is modified in-place or replaced before `prepare_request`.
    # But `prepare_request` is called with the modified messages.
    # Since we mocked `send_async`, `prepare_request` (a real method) was called.
    # `prepare_request` for OpenAI converts to dicts.
    
    # Let's check expected content in the payload sent to "network"
    last_msg = sent_messages[0] # The tool message (Anthropic/OpenAI formatted)
    
    # OpenAI format: {"role": "tool", "content": ...}
    # We used "gpt-4o" so it should use OpenAIProvider
    print(f"DEBUG: Sent message structure: {last_msg}")
    
    content = last_msg["content"]
    if "(trimmed)" in content:
        print(f"✅ Content was trimmed: '{content}'")
    else:
        print(f"❌ Content was NOT trimmed: '{content}'")

if __name__ == "__main__":
    asyncio.run(run_verification())
