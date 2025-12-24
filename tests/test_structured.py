import json
from pydantic import BaseModel
from aiclient.providers.openai import OpenAIProvider
from aiclient.data_types import UserMessage

class UserInfo(BaseModel):
    name: str
    age: int

def test_openai_structured_output_payload():
    provider = OpenAIProvider(api_key="test")
    messages = [UserMessage(content="Hello")]
    schema = UserInfo.model_json_schema()
    
    url, data = provider.prepare_request(
        "gpt-4o", 
        messages, 
        response_schema=schema, 
        strict=True
    )
    
    # Check if response_format is correctly populated
    assert "response_format" in data
    rf = data["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema
    # Schema key order checking might be tricky if dict not predictable, but == checks contents usually.

def test_openai_structured_output_payload_no_strict():
    provider = OpenAIProvider(api_key="test")
    messages = [UserMessage(content="Hello")]
    schema = UserInfo.model_json_schema()
    
    # Even if strict=False, if passed to provider, it sets it.
    # ChatModel logic handles conditional passing.
    # But if we pass it manually:
    url, data = provider.prepare_request(
        "gpt-4o", 
        messages, 
        response_schema=schema, 
        strict=False
    )
    
    assert "response_format" in data
    assert data["response_format"]["json_schema"]["strict"] is False
