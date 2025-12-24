"""
Tests for aiclient.memory module.
"""
import pytest
from aiclient.memory import ConversationMemory, SlidingWindowMemory
from aiclient.types import UserMessage, SystemMessage, AssistantMessage

def test_conversation_memory():
    """Test basic storage and retrieval."""
    mem = ConversationMemory()
    mem.add_message(UserMessage(content="Hello"))
    mem.add_message(AssistantMessage(content="Hi"))
    
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "Hello"
    assert msgs[1].content == "Hi"

def test_sliding_window_memory_truncation():
    """Test standard truncation."""
    # Max 2 messages
    mem = SlidingWindowMemory(max_messages=2)
    
    mem.add_message(UserMessage(content="1"))
    mem.add_message(AssistantMessage(content="2"))
    mem.add_message(UserMessage(content="3"))
    
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "2"
    assert msgs[1].content == "3"

def test_sliding_window_memory_preserves_system():
    """Test system message preservation."""
    # Max 3 messages (System + 2 others)
    mem = SlidingWindowMemory(max_messages=3)
    
    mem.add_message(SystemMessage(content="System"))
    mem.add_message(UserMessage(content="1"))
    mem.add_message(AssistantMessage(content="2"))
    mem.add_message(UserMessage(content="3")) 
    
    msgs = mem.get_messages()
    assert len(msgs) == 3
    assert isinstance(msgs[0], SystemMessage)
    # 2 and 3 should remain. 1 is evicted.
    assert msgs[1].content == "2"
    assert msgs[2].content == "3"

def test_memory_serialization():
    """Test save/load."""
    mem = ConversationMemory()
    mem.add_message(UserMessage(content="Save me"))
    
    data = mem.save()
    assert data["messages"][0]["content"] == "Save me"
    
    mem2 = ConversationMemory()
    mem2.load(data)
    assert len(mem2.get_messages()) == 1
    assert mem2.get_messages()[0].content == "Save me"
