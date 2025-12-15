import pytest
from aiclient.client import Client
from aiclient.tools.base import Tool
from aiclient.tools.policy import policy_tool
from aiclient.agents.simple import SimpleAgent
from pydantic import BaseModel

def test_imports():
    assert Client
    assert Tool
    assert SimpleAgent

def test_tool_execution():
    assert policy_tool.name == "check_policy"
    result = policy_tool.run(text="This is safe content")
    assert result is True
    
    result = policy_tool.run(text="This contains forbidden content")
    assert result is False

def test_agent_instantiation():
    client = Client(openai_api_key="sk-test")
    agent = SimpleAgent(client, "gpt-4", tools=[policy_tool])
    assert agent
    assert len(agent.tools) == 1
