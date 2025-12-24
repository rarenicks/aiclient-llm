"""
Example: Testing your AI Application
Run: pytest examples/testing_your_app.py
"""
import pytest
from aiclient import Client
from aiclient.testing import MockProvider

# --- Your Application Code ---
class ChatBot:
    def __init__(self, client: Client):
        self.client = client

    def greet(self, name: str) -> str:
        response = self.client.chat("gpt-4o").generate(f"Greet {name}")
        return response.text

# --- Your Tests ---
def test_chatbot_greeting():
    # 1. Setup Mock
    mock_provider = MockProvider()
    mock_provider.add_response("Hello, Alice! Welcome.")
    
    # 2. Setup Client with Mock Injection
    client = Client()
    
    # Monkey-patching for this test. 
    # In a real app, you might use dependency injection or a factory.
    # Here we intercept proper model creation to inject our mock provider.
    original_chat = client.chat
    def mock_chat(model):
        m = original_chat(model)
        m.provider = mock_provider
        # IMPORTANT: Swap transport to avoid making real HTTP requests
        from aiclient.testing import MockTransport
        m.transport = MockTransport() 
        return m
    client.chat = mock_chat

    # 3. Test App
    bot = ChatBot(client)
    reply = bot.greet("Alice")

    # 4. Assertions
    assert reply == "Hello, Alice! Welcome."
    
    # Verify request payload
    last_request = mock_provider.requests[0]
    assert last_request["model"] == "gpt-4o"
    assert "Alice" in last_request["messages"][0]["content"]
    
    print("✅ Test Passed!")

if __name__ == "__main__":
    test_chatbot_greeting()
