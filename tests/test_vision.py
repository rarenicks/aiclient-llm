import pytest
import base64
from unittest.mock import patch, mock_open, MagicMock
from aiclient.data_types import Image, UserMessage, Text
from aiclient.providers.openai import OpenAIProvider
from aiclient.providers.anthropic import AnthropicProvider
from aiclient.providers.google import GoogleProvider

# --- Image Class Tests ---

def test_image_from_base64():
    img = Image(base64_data="abc", media_type="image/png")
    assert img.to_base64() == "abc"

def test_image_from_path():
    with patch("builtins.open", mock_open(read_data=b"fake_image_data")):
        with patch("pathlib.Path.exists", return_value=True):
            img = Image(path="test.png")
            # b64(fake_image_data) -> ZmFrZV9pbWFnZV9kYXRh
            assert img.to_base64() == "ZmFrZV9pbWFnZV9kYXRh"

def test_image_from_url():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.content = b"fake_url_data"
        mock_get.return_value.raise_for_status = MagicMock()
        
        img = Image(url="http://example.com/img.jpg")
        # b64(fake_url_data) -> ZmFrZV91cmxfZGF0YQ==
        assert img.to_base64() == "ZmFrZV91cmxfZGF0YQ=="

# --- Provider Format Tests ---

def test_openai_vision_format():
    p = OpenAIProvider(api_key="sk-test")
    img = Image(base64_data="abc", media_type="image/png")
    msg = UserMessage(content=[Text(text="Look"), img])
    
    # We mock to_base64 just to be safe or rely on logic
    # Since we passed base64_data, it returns it directly.
    
    _, data = p.prepare_request("gpt-4o", [msg])
    
    content = data["messages"][0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Look"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,abc"

def test_anthropic_vision_format():
    p = AnthropicProvider(api_key="sk-test")
    img = Image(base64_data="abc", media_type="image/jpeg")
    msg = UserMessage(content=[Text(text="Look"), img])
    
    _, data = p.prepare_request("claude-3-opus", [msg])
    
    content = data["messages"][0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Look"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["data"] == "abc"

def test_google_vision_format():
    p = GoogleProvider(api_key="key")
    img = Image(base64_data="abc", media_type="image/jpeg")
    msg = UserMessage(content=[Text(text="Look"), img])
    
    _, data = p.prepare_request("gemini-1.5-pro", [msg])
    
    parts = data["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0] == {"text": "Look"}
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert parts[1]["inlineData"]["data"] == "abc"
