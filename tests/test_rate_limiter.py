
import asyncio
import time
import pytest
from unittest.mock import MagicMock
from aiclient.resilience import RateLimiter
from aiclient.data_types import UserMessage

# Helper to estimate tokens (same as implementation)
def estimate_tokens(text):
    return len(text) // 4

def test_rpm_sync():
    """Test standard RPM limiting in sync mode."""
    # Limit to 2 requests per minute (very strict for testing)
    limiter = RateLimiter(requests_per_minute=2)
    
    start = time.time()
    limiter.before_request("model", "req1")
    limiter.before_request("model", "req2")
    
    # Third request should block until window clears? 
    # Wait, window is 60s. That's too long for unit test.
    # We should mock time or patch the window.
    limiter.window = 1.0 # 1 second window
    
    limiter.before_request("model", "req3")
    end = time.time()
    
    # Should have waited at least a bit if the first two were instant?
    # Actually, simplistic implementation:
    # req1: t=0.0
    # req2: t=0.0 (approx)
    # req3: needs to wait until t=1.0?
    
    # Since we sleep inside, 'end - start' should be >= 1.0 if we hit limit?
    # But only if we hit the limit which we did (limit 2, sent 3).
    # Since we set window=1.0, and sent 3 immediately.
    # The 3rd one should wait.
    assert (end - start) >= 0.9 # Tolerance

@pytest.mark.asyncio
async def test_rpm_async():
    """Test RPM limiting in async mode (should verify it doesn't block effectively, but difficult to test strictly without mock time)."""
    limiter = RateLimiter(requests_per_minute=2)
    limiter.window = 0.5
    
    start = time.time()
    await limiter.before_request_async("model", "req1")
    await limiter.before_request_async("model", "req2")
    await limiter.before_request_async("model", "req3")
    end = time.time()
    
    assert (end - start) >= 0.45

def test_tpm_sync():
    """Test TPM limiting."""
    # 20 tokens per minute
    limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=20)
    limiter.window = 1.0
    
    # "Hello" is 5 chars => 1 token approx
    # Let's send a big request.
    prompt = "x" * 40 # 10 tokens
    
    start = time.time()
    limiter.before_request("model", prompt) # 10 tokens used
    limiter.before_request("model", prompt) # 20 tokens used (limit reached)
    
    # Next one should block
    limiter.before_request("model", "short") # 1 token
    end = time.time()
    
    assert (end - start) >= 0.9

@pytest.mark.asyncio
async def test_tpm_async():
    """Test TPM limiting async."""
    limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=10)
    limiter.window = 0.5
    
    prompt = "x" * 40 # 10 tokens. Hits limit immediately.
    
    start = time.time()
    await limiter.before_request_async("model", prompt)
    
    # Next request must wait
    await limiter.before_request_async("model", "xxxx")
    end = time.time()
    
    assert (end - start) >= 0.45 
