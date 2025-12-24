class AIClientError(Exception):
    """Base exception for all aiclient errors."""
    pass

class AuthenticationError(AIClientError):
    """Raised when API credentials are invalid (401/403)."""
    pass

class RateLimitError(AIClientError):
    """Raised when API rate limit is exceeded (429)."""
    pass

class ProviderError(AIClientError):
    """Raised when the provider returns a 5xx error or other API failure."""
    pass

class InvalidRequestError(AIClientError):
    """Raised when the request is malformed (400)."""
    pass

class NetworkError(AIClientError):
    """Raised when network connection fails."""
    pass
