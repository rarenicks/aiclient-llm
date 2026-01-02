# Embeddings 🔢

`aiclient-llm` provides a unified interface for generating vector embeddings from text. This is essential for RAG (Retrieval Augmented Generation), semantic search, and clustering.

## Supported Providers
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Google**: `text-embedding-004`
- **xAI**: `grok-beta` (if embedding supported)
- **Ollama**: Any loaded embedding model (e.g. `nomic-embed-text`)

## API Reference

The client provides two main methods:

### `client.embed(text, model)`

Generate a single embedding vector.

```python
from aiclient import Client

client = Client()

vector = await client.embed(
    "The quick brown fox jumps over the lazy dog",
    model="text-embedding-3-small"
)

print(f"Vector length: {len(vector)}") # e.g., 1536
print(vector[:5])
```

### `client.embed_batch(texts, model)`

Generate embeddings for a list of texts efficiently. The client handles batching logic required by the provider.

```python
texts = [
    "Deep learning transforms fields",
    "Natural language processing is growing",
    "AI agents are the future"
]

vectors = await client.embed_batch(
    texts, 
    model="text-embedding-004"
)

print(f"Generated {len(vectors)} vectors")
```

## Usage with Semantic Cache

Embeddings are the backbone of the `SemanticCacheMiddleware`, which allows you to cache responses for similar questions.

```python
from aiclient import SemanticCacheMiddleware

# Create a simple adapter that conforms to the embedding interface
class MyEmbedder:
    def embed(self, text):
        return client.embed(text, "text-embedding-3-small")

client.add_middleware(SemanticCacheMiddleware(
    embedder=MyEmbedder(),
    threshold=0.9
))
```
