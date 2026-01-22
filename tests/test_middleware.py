from aiclient import Client
from aiclient.middleware import CostTrackingMiddleware


# Mocking infrastructure
class MockResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class MockTransport:
    def __init__(self, response_data=None, **kwargs):
        self.response_data = response_data or {}
        self.sent_data = []

    def send(self, endpoint, data):
        self.sent_data.append((endpoint, data, None))
        return self.response_data


class SuffixMiddleware:
    def before_request(self, model, prompt):
        # prompt is List[BaseMessage]
        last_msg = prompt[-1]
        last_msg.content += " [SUFFIX]"
        return prompt

    def after_response(self, response):
        return response

    def on_error(self, error, model):
        pass


def test_middleware_modifies_request():
    # Setup Client
    client = Client(
        openai_api_key="sk-test",
        transport_factory=lambda **kwargs: MockTransport(
            **kwargs,
            response_data={
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ),
    )

    # Add Middleware
    client.add_middleware(SuffixMiddleware())

    # Chat
    model = client.chat("gpt-4")
    model.generate("Hello")

    # Verify Middleware ran (by checking what was sent)
    # The MockTransport stores sent_data
    sent_data = model.transport.sent_data
    assert len(sent_data) > 0
    last_req = sent_data[-1]
    url, data, headers = last_req

    messages = data["messages"]
    assert messages[0]["content"] == "Hello [SUFFIX]"


def test_cost_tracking_middleware():
    client = Client(
        anthropic_api_key="sk-test",
        transport_factory=lambda **kwargs: MockTransport(
            **kwargs,
            response_data={
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "output_tokens": 5,
                    "input_tokens": 15,
                    "cache_read_input_tokens": 25,
                    "cache_creation_input_tokens": 35,
                },
            },
        ),
    )

    tracker = CostTrackingMiddleware()
    client.add_middleware(tracker)

    model = client.chat("claude-opus-4.5")
    model.generate("Hello")

    assert tracker.total_cache_creation_input_tokens == 35
    assert tracker.total_cache_read_input_tokens == 25
    # Anthropic's `input_tokens` does not include `cache_read_input_tokens`
    # So total = input + cache_read
    assert tracker.total_input_tokens == 40
    assert tracker.total_output_tokens == 5
