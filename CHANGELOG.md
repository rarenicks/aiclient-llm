# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-12-24 ("The Adoption Layer")

### 🚀 New Features

- **Streaming Support**: Real-time token streaming with `model.stream()` and examples.
- **Memory System**: `ConversationMemory` and `SlidingWindowMemory` for managing chat context.
- **Testing Utilities**: `MockProvider` and `MockTransport` for deterministic testing.
- **Async Batch Processing**: `Client.batch()` and `BatchProcessor` for concurrent requests.
- **Multimodal (Vision)**: Unified `Image` type supporting paths, URLs, and Base64.
- **Model Context Protocol (MCP)**: Support for connecting to external tools via MCP.
- **Semantic Caching**: Embedding-based response caching `SemanticCacheMiddleware`.
- **Resilience**: `RetryMiddleware` (backoff/jitter), `CircuitBreaker`, `RateLimiter`, and `FallbackChain`.
- **Structured Outputs**: Native support for strict JSON Schemas (OpenAI).
- **Observability**: `TracingMiddleware` and `OpenTelemetryMiddleware` hooks.
- **Debug Mode**: Verbose logging enabled via `Client(debug=True)`.
- **Enhanced Error Handling**: Typed exceptions (`AuthenticationError`, `RateLimitError`, `NetworkError`, etc.).

### ⚡ Improvements

- **Type Safety**: Comprehensive type hints across `Client`, `BatchProcessor`, and `Agent`.
- **Performance**: Cached tool lookups in `MCPServerManager`.
- **Usage Tracking**: Enhanced metrics for cache hits and costs.
- **Midleware Hooks**: Added `on_error` support to middleware protocol.

### 📚 Documentation & Examples

- **Production Cookbook**: 5+ real-world examples (Memory, RAG, Agents, Batching).
- **New Guides**: Testing, Memory, Error Handling, and Streaming documentation.
- **Updated README**: Verified badges and comprehensive feature list.

### 🐛 Bug Fixes

- Fixed `httpx` streaming error handling (properly raises `AuthenticationError`).
- Fixed `Memory.load()` serialization.
- Renamed `ConnectionError` to `NetworkError`.
- Fixed `MCPServerManager` initialization.

---
