
import time
import asyncio
from unittest.mock import MagicMock, AsyncMock
from aiclient.client import Client
from aiclient.resilience import RateLimiter

def test_sync_rate_limiter():
    print("\n--- Testing Sync Rate Limiter (RPM=60, TPM=20) ---")
    # TPM=20. "hello" is 5 chars ~ 1 token.
    # We set RPM high so TPM is the bottleneck.
    limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=20)
    # Hack window to 1s for testing speed
    limiter.window = 1.0
    
    client = Client(openai_api_key="mock")
    client.add_middleware(limiter)
    
    # Mock transport
    mock_resp = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    client.transport_factory = MagicMock()
    transport_instance = client.transport_factory.return_value
    transport_instance.send.return_value = mock_resp
    
    model = client.chat("gpt-4o")
    
    # Send 25 tokens worth of requests.
    # Request 1: 10 tokens ("x"*40)
    # Request 2: 10 tokens ("x"*40) -> Total 20. Limit reached.
    # Request 3: 5 tokens ("x"*20) -> Should block.
    
    prompt_10 = "x" * 40
    prompt_5 = "x" * 20
    
    start = time.time()
    print("Sending Req 1 (10 tokens)...")
    model.generate(prompt_10)
    
    print("Sending Req 2 (10 tokens)...")
    model.generate(prompt_10)
    
    print("Sending Req 3 (5 tokens) - Should Wait...")
    model.generate(prompt_5)
    end = time.time()
    
    duration = end - start
    print(f"Total duration: {duration:.2f}s")
    
    if duration >= 0.9:
        print("✅ Rate Limiter enforced delay (Sync)")
    else:
        print("❌ Rate Limiter FAILED to enforce delay (Sync)")

async def test_async_rate_limiter():
    print("\n--- Testing Async Rate Limiter (RPM=100, TPM=10) ---")
    # TPM=10. Window=0.5s
    limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=10)
    limiter.window = 0.5
    
    client = Client(openai_api_key="mock")
    client.add_middleware(limiter)
    
    # Mock transport
    mock_resp = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    client.transport_factory = MagicMock()
    transport_instance = client.transport_factory.return_value
    transport_instance.send_async = AsyncMock(return_value=mock_resp)
    
    model = client.chat("gpt-4o")
    
    prompt_10 = "x" * 40 # 10 tokens
    
    start = time.time()
    print("Sending Req 1 (10 tokens) - Hits limit...")
    await model.generate_async(prompt_10)
    
    print("Sending Req 2 (1 token) - Should Wait...")
    await model.generate_async("x"*4)
    end = time.time()
    
    duration = end - start
    print(f"Total duration: {duration:.2f}s")
    
    if duration >= 0.45:
        print("✅ Rate Limiter enforced delay (Async)")
    else:
        print("❌ Rate Limiter FAILED to enforce delay (Async)")

if __name__ == "__main__":
    test_sync_rate_limiter()
    asyncio.run(test_async_rate_limiter())
