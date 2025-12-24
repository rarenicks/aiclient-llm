import pytest
from unittest.mock import MagicMock
from aiclient.resilience import FallbackChain, LoadBalancer
from aiclient.data_types import ModelResponse

class MockClient:
    def __init__(self):
        self.chat_mock = MagicMock()
        
    def chat(self, model: str):
        return self.chat_mock(model)

def test_fallback_chain_success():
    client = MockClient()
    # First model works
    model_mock = MagicMock()
    model_mock.generate.return_value = ModelResponse(text="Success", raw={}, provider="m1")
    client.chat_mock.return_value = model_mock
    
    chain = FallbackChain(client, ["m1", "m2"])
    resp = chain.generate("test")
    
    assert resp.text == "Success"
    # Should verify client.chat called with m1 
    # (Mock logic is a bit circular here, simpler to trust behavior or use side_effect)

def test_fallback_chain_partial_failure():
    client = MockClient()
    
    # Setup: m1 fails, m2 succeeds
    m1_mock = MagicMock()
    m1_mock.generate.side_effect = Exception("M1 Failed")
    
    m2_mock = MagicMock()
    m2_mock.generate.return_value = ModelResponse(text="M2 Success", raw={}, provider="m2")
    
    def side_effect(model):
        if model == "m1": return m1_mock
        if model == "m2": return m2_mock
        return MagicMock()
        
    client.chat_mock.side_effect = side_effect
    
    chain = FallbackChain(client, ["m1", "m2"])
    resp = chain.generate("test")
    
    assert resp.text == "M2 Success"

def test_load_balancer_round_robin():
    client = MockClient()
    # Setup mocks
    mock_gen = MagicMock()
    mock_gen.generate.return_value = ModelResponse(text="OK", raw={}, provider="any")
    
    client.chat_mock.return_value = mock_gen
    
    lb = LoadBalancer(client, ["m1", "m2"])
    
    # 1st call -> m1
    lb.generate("test")
    args1, _ = client.chat_mock.call_args
    # client.chat_mock("m1") was called?
    # Because client.chat_mock is a Mock, call_args stores last call.
    # But we called generate(), which called chat(model).generate().
    # Checking chat call args:
    assert client.chat_mock.call_args[0][0] == "m1"
    
    # 2nd call -> m2
    lb.generate("test")
    assert client.chat_mock.call_args[0][0] == "m2"
    
    # 3rd call -> m1
    lb.generate("test")
    assert client.chat_mock.call_args[0][0] == "m1"
