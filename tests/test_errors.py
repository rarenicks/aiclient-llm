"""
Tests for error mapping in HTTPTransport.
"""
import pytest
import httpx
from aiclient.transport.http import HTTPTransport
from aiclient.exceptions import (
    AuthenticationError, 
    RateLimitError, 
    ProviderError, 
    InvalidRequestError
)
from unittest.mock import MagicMock, patch

def test_error_mapping_401():
    transport = HTTPTransport()
    
    # Mock send to raise HTTPStatusError
    with patch.object(transport.client, "post") as mock_post:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        # httpx raises this if raise_for_status called
        def raise_for_status():
            raise httpx.HTTPStatusError("401 Unauthorized", request=MagicMock(), response=mock_response)
            
        mock_response.raise_for_status = raise_for_status
        mock_post.return_value = mock_response
        
        with pytest.raises(AuthenticationError, match="Unauthorized"):
            transport.send("http://test", {})

def test_error_mapping_429():
    transport = HTTPTransport()
    
    with patch.object(transport.client, "post") as mock_post:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        
        def raise_for_status():
            raise httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=mock_response)
        
        mock_response.raise_for_status = raise_for_status
        mock_post.return_value = mock_response
        
        with pytest.raises(RateLimitError, match="Too Many Requests"):
            transport.send("http://test", {})

def test_error_mapping_500():
    transport = HTTPTransport()
    
    with patch.object(transport.client, "post") as mock_post:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        
        def raise_for_status():
            raise httpx.HTTPStatusError("503 Service Unavailable", request=MagicMock(), response=mock_response)
        
        mock_response.raise_for_status = raise_for_status
        mock_post.return_value = mock_response
        
        with pytest.raises(ProviderError, match="Service Unavailable"):
            transport.send("http://test", {})
