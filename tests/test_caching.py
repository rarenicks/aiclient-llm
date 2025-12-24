import pytest
from aiclient.providers.anthropic import AnthropicProvider
from aiclient.data_types import SystemMessage, UserMessage, Text

def test_anthropic_caching_injection():
    provider = AnthropicProvider(api_key="test")
    
    # 1. Test System Message Caching
    messages = [
        SystemMessage(content="System Prompt", cache_control="ephemeral")
    ]
    endpoint, payload = provider.prepare_request("claude-3-opus", messages)
    
    system_val = payload["system"]
    assert isinstance(system_val, list)
    assert system_val[0]["type"] == "text"
    assert system_val[0]["cache_control"] == {"type": "ephemeral"}

    # 2. Test User Message Caching
    messages = [
        UserMessage(content="Hello", cache_control="ephemeral")
    ]
    
    endpoint, payload = provider.prepare_request("claude-3-opus", messages)
    
    # Check User message structure
    user_msg_content = payload["messages"][0]["content"]
    assert isinstance(user_msg_content, list)
    assert len(user_msg_content) == 1
    assert user_msg_content[0]["type"] == "text"
    assert user_msg_content[0]["text"] == "Hello"
    assert user_msg_content[0]["cache_control"] == {"type": "ephemeral"}

def test_anthropic_multi_block_caching():
    provider = AnthropicProvider(api_key="test")
    
    messages = [
        UserMessage(
            content=[
                Text(text="Context 1"),
                Text(text="Context 2")
            ],
            cache_control="ephemeral"
        )
    ]
    
    endpoint, payload = provider.prepare_request("claude-3-opus", messages)
    content = payload["messages"][0]["content"]
    
    # Should only apply to the LAST block
    assert "cache_control" not in content[0]
    assert content[1]["cache_control"] == {"type": "ephemeral"}
