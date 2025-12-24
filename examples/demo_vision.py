import asyncio
import os
import sys
from aiclient import Client
from aiclient.data_types import UserMessage, Text, Image

# 1x1 Red Dot
RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

async def test_provider(client, model, provider_name):
    print(f"Testing {provider_name} ({model})...", end=" ", flush=True)
    try:
        msg = UserMessage(content=[
            Text(text="What color is this single pixel image? Answer in one word."),
            Image(base64_data=RED_DOT_B64, media_type="image/png")
        ])
        
        response = await client.chat(model).generate_async([msg])
        ans = response.text.lower()
        if "red" in ans:
            print(f"✅ ({response.text})")
            return True
        else:
            print(f"⚠️ (Response: {response.text})")
            return False
            
    except Exception as e:
        print(f"❌ {e}")
        if hasattr(e, "response"):
            print(f"Body: {e.response.text[:200]}")
        return False

async def main():
    client = Client()
    
    print("🤖 VISION VERIFICATION")
    print("=======================")
    
    # OpenAI
    await test_provider(client, "gpt-4o", "OpenAI")
    
    # Anthropic
    await test_provider(client, "claude-3-haiku-20240307", "Anthropic") # Opus supports vision
    
    # Google
    await test_provider(client, "gemini-2.0-flash-exp", "Google")
    
    # xAI (Grok-2 Vision)
    await test_provider(client, "grok-2-vision-latest", "xAI")

if __name__ == "__main__":
    asyncio.run(main())
