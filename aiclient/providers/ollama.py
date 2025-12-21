from .openai import OpenAIProvider

class OllamaProvider(OpenAIProvider):
    def __init__(self, api_key: str = "ollama", base_url: str = "http://localhost:11434/v1"):
        super().__init__(api_key=api_key, base_url=base_url)
