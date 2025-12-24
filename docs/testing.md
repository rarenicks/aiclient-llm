# Testing Your Application 🧪

`aiclient` provides built-in utilities to help you test your AI-powered applications deterministically, without incurring costs or latency from real API calls.

## The MockProvider

The `MockProvider` mimics a real LLM provider but returns static, pre-defined responses.

### Basic Usage

```python
from aiclient import Client
from aiclient.testing import MockProvider, MockTransport

def test_my_chatbot_logic():
    # 1. Setup Mock
    mock_provider = MockProvider()
    mock_provider.add_response("Hello, human!")
    
    # 2. Inject into Client
    # Currently, Client instantiates providers internally based on name.
    # To use a mock, we can manually attach it or use dependency injection in your app.
    
    client = Client()
    # HACK: Manually override the provider for the model
    model = client.chat("gpt-4o")
    model.provider = mock_provider
    model.transport = MockTransport() # Bypass HTTP layer
    
    # 3. Run your code
    response = model.generate("Hi")
    
    # 4. Verify
    assert response.text == "Hello, human!"
    assert len(mock_provider.requests) == 1
    assert mock_provider.requests[0]["messages"][0].content == "Hi"
```

## Testing Agents

You can extensively test Agent tool loops by queuing multiple responses.

```python
from aiclient.testing import MockProvider
from aiclient.types import ModelResponse, ToolCall

def test_agent_flow():
    mock = MockProvider()
    
    # Response 1: Call a tool
    mock.add_response(
        text="Let me check weather",
        raw={}, 
        # You'd need to mock the ModelResponse structure carefully if you want full tool support
        # or use helper methods if available.
    )
    # ...
```

## Integration Testing (Pytest Fixture)

We recommend creating a pytest fixture for your client.

```python
# conftest.py
import pytest
from aiclient import Client
from aiclient.testing import MockProvider

@pytest.fixture
def mock_client():
    client = Client()
    client.mock_provider = MockProvider()
    # Patch client.chat to always return a model using this mock provider
    original_chat = client.chat
    
    def mock_chat(model_name):
        model = original_chat(model_name)
        model.provider = client.mock_provider
        return model
        
    client.chat = mock_chat
    return client
```
