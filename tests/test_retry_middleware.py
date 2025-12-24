import pytest
import asyncio
import time
import httpx
from unittest.mock import MagicMock, call, patch, AsyncMock
from aiclient.resilience.retries import RetryMiddleware

# Mock Request/Response for httpx
def mock_httpx_error(status_code):
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("Error", request=request, response=response)

def test_should_retry_logic():
    mw = RetryMiddleware()
    
    # 429 -> Retry
    assert mw.should_retry(mock_httpx_error(429)) is True
    # 500, 503 -> Retry
    assert mw.should_retry(mock_httpx_error(500)) is True
    assert mw.should_retry(mock_httpx_error(503)) is True
    
    # 400, 404 -> No Retry
    assert mw.should_retry(mock_httpx_error(400)) is False
    assert mw.should_retry(mock_httpx_error(404)) is False
    
    # Random exception -> No Retry
    assert mw.should_retry(ValueError("bad")) is False

def test_on_error_sync_retryable():
    mw = RetryMiddleware(max_retries=3, backoff_factor=0.1)
    error = mock_httpx_error(500)
    
    with patch("time.sleep") as mock_sleep:
        # Attempt 0
        mw.on_error(error, "model", attempt=0)
        mock_sleep.assert_called_once()
        # Delay should be around 0.1 * 2^0 = 0.1
        delay = mock_sleep.call_args[0][0]
        assert 0.1 <= delay <= 0.15 # allowing for jitter

def test_on_error_sync_non_retryable():
    mw = RetryMiddleware()
    error = mock_httpx_error(400)
    
    # Should raise error immediately
    with pytest.raises(httpx.HTTPStatusError):
        mw.on_error(error, "model", attempt=0)

def test_on_error_sync_exhausted():
    mw = RetryMiddleware(max_retries=3)
    error = mock_httpx_error(500)
    
    # If attempt >= max_retries, should raise
    with pytest.raises(httpx.HTTPStatusError):
        mw.on_error(error, "model", attempt=3)

@pytest.mark.asyncio
async def test_on_error_async():
    mw = RetryMiddleware(max_retries=3, backoff_factor=0.1)
    error = mock_httpx_error(503)
    
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await mw.on_error_async(error, "model", attempt=1)
        
        mock_sleep.assert_awaited_once()
        delay = mock_sleep.call_args[0][0]
        # 0.1 * 2^1 = 0.2
        assert 0.2 <= delay <= 0.3

@pytest.mark.asyncio
async def test_on_error_async_exhausted():
    mw = RetryMiddleware(max_retries=3)
    error = mock_httpx_error(503)
    
    with pytest.raises(httpx.HTTPStatusError):
        await mw.on_error_async(error, "model", attempt=5)
