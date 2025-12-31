import numpy as np
from typing import List, Optional, Any, Dict, Protocol
from ..middleware import Middleware
from ..data_types import BaseMessage, ModelResponse, UserMessage

class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
        ...

class VectorStore(Protocol):
    def add(self, vector: List[float], value: Any):
        ...
    def search(self, vector: List[float], threshold: float) -> Optional[Any]:
        ...

class InMemoryVectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.values: List[Any] = []

    def add(self, vector: List[float], value: Any):
        self.vectors.append(np.array(vector))
        self.values.append(value)

    def search(self, vector: List[float], threshold: float) -> Optional[Any]:
        if not self.vectors:
            return None
        
        query = np.array(vector)
        # Cosine similarity
        # Setup: norm(query) * norm(vec)
        q_norm = np.linalg.norm(query)
        
        best_score = -1.0
        best_idx = -1
        
        for i, vec in enumerate(self.vectors):
            score = np.dot(query, vec) / (q_norm * np.linalg.norm(vec))
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_score >= threshold:
            return self.values[best_idx]
        return None

class SemanticCacheMiddleware(Middleware):
    def __init__(
        self, 
        embedder: EmbeddingProvider, 
        threshold: float = 0.9,
        backend: Optional[VectorStore] = None
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.store = backend or InMemoryVectorStore()
        self._last_prompt_str = None

    def before_request(self, model: str, prompt: Any) -> Any:
        # 1. Extract text from prompt (assume last user message or string)
        text = ""
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list):
            # Find last user message
            for m in reversed(prompt):
                if isinstance(m, UserMessage):
                    text = str(m.content) # Simple string conversion
                    break
        
        if not text:
            return prompt

        self._last_prompt_str = text
        
        # 2. Embed
        vector = self.embedder.embed(text)
        
        # 3. Search
        cached_response = self.store.search(vector, self.threshold)
        if cached_response:
            # Short-circuit by returning response
            # Note: cached_response is stored as ModelResponse
            if isinstance(cached_response, ModelResponse):
                # Update usage to indicate cache hit?
                # Maybe set a flag
                return cached_response
            
        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        # Cache the response for the last prompt
        if self._last_prompt_str:
            vector = self.embedder.embed(self._last_prompt_str)
            self.store.add(vector, response)
        return response
