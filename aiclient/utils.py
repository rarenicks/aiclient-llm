import base64
import mimetypes
import os
from typing import Tuple
from .types import Image

def encode_image(image: Image) -> Tuple[str, str]:
    """
    Returns (media_type, base64_data).
    Resolves path -> base64 or returns existing base64/url.
    """
    if image.base64_data:
        return image.media_type, image.base64_data
    
    if image.url:
        # For URLs, we usually pass them directly to provider if supported.
        # But if we need to download, we would do it here. 
        # For now, return empty base64 and let provider handle URL logic if it can, 
        # or provider handles the Image object directly.
        # Actually, let's just helper for local files.
        return image.media_type, ""

    if image.path:
        if not os.path.exists(image.path):
            raise FileNotFoundError(f"Image not found: {image.path}")
        
        mime_type, _ = mimetypes.guess_type(image.path)
        media_type = mime_type or "image/jpeg"
        
        with open(image.path, "rb") as f:
            return media_type, base64.b64encode(f.read()).decode("utf-8")
            
    raise ValueError("Image must have path, url, or base64_data")

def should_retry(exception: Exception) -> bool:
    """Check if the exception is a retryable HTTP error (429, 5xx)."""
    if hasattr(exception, "response"):
        code = exception.response.status_code
        # 429: Too Many Requests
        # 5xx: Server Errors (500, 502, 503, 504)
        if code == 429 or 500 <= code < 600:
            return True
    return False
