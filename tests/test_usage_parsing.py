from aiclient.providers.google import GoogleProvider
from aiclient.providers.openai import OpenAIProvider


def test_openai_parsing():
    provider = OpenAIProvider(api_key="mock_key")

    # Mock response data with cached tokens
    response_data = {
        "choices": [{"message": {"content": "Hello"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {
                "cached_tokens": 25
            }
        }
    }

    model_response = provider.parse_response(response_data)

    usage = model_response.usage

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.cache_read_input_tokens == 25

def test_google_parsing():
    provider = GoogleProvider(api_key="mock_key")

    # Mock response data with cached tokens
    response_data = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
            "cachedContentTokenCount": 30
        }
    }

    model_response = provider.parse_response(response_data)

    usage = model_response.usage

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.cache_read_input_tokens == 30
