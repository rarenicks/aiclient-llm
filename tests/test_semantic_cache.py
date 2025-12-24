import pytest
import numpy as np
from typing import List
from aiclient.cache.semantic import SemanticCacheMiddleware, InMemoryVectorStore
from aiclient.data_types import ModelResponse, UserMessage, Usage

class MockEmbedder:
    def embed(self, text: str) -> List[float]:
        # Simple deterministic embedding
        if "hello" in text:
            return [1.0, 0.0]
        if "bye" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

def test_vector_store():
    store = InMemoryVectorStore()
    store.add([1.0, 0.0], "response1")
    store.add([0.0, 1.0], "response2")
    
    # Exact match
    assert store.search([1.0, 0.0], threshold=0.99) == "response1"
    
    # Orthogonal
    assert store.search([0.0, 1.0], threshold=0.99) == "response2"
    
    # No match
    assert store.search([0.5, 0.5], threshold=0.99) is None

def test_semantic_middleware_hit():
    embedder = MockEmbedder()
    mw = SemanticCacheMiddleware(embedder, threshold=0.9)
    
    # 1. Seed cache
    resp = ModelResponse(
        text="Cached Hello", 
        raw={}, 
        provider="mock",
        usage=Usage()
    )
    # Manually seed for test
    mw.store.add([1.0, 0.0], resp)
    
    # 2. Request that matches "intently"
    prompt = "hello world"
    result = mw.before_request("model", prompt)
    
    # Should return SHORT-CIRCUIT response
    assert isinstance(result, ModelResponse)
    assert result.text == "Cached Hello"

def test_semantic_middleware_miss():
    embedder = MockEmbedder()
    mw = SemanticCacheMiddleware(embedder, threshold=0.9)
    
    # Request "bye" - nothing in cache
    prompt = "bye"
    result = mw.before_request("model", prompt)
    
    # Should pass through prompt
    assert isinstance(result, str)
    assert result == "bye"
    
    # After response, should cache
    resp = ModelResponse(text="Bye Response", raw={}, provider="mock")
    mw.after_response(resp)
    
    # Now verify it's in store (MockEmbedder returns [0.0, 1.0] for "bye")
    assert len(mw.store.vectors) == 1
    assert np.array_equal(mw.store.vectors[0], np.array([0.0, 1.0]))
