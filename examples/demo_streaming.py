"""
Demo: Streaming Responses in Console
Run: python examples/demo_streaming.py
"""
import os
import asyncio
from dotenv import load_dotenv
from aiclient import Client

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env")
        return

    client = Client()
    print("🤖: I'm ready! (Sync Streaming)")
    
    # Sync Streaming
    print("User: Count to 10 quickly.")
    print("AI: ", end="", flush=True)
    
    for chunk in client.chat("gpt-4o-mini").stream("Count to 10 quickly."):
        print(chunk, end="", flush=True)
    print("\n")

async def main_async():
    client = Client()
    print("🤖: Async mode activating! (Async Streaming)")
    
    print("User: Write a timber haiku.")
    print("AI: ", end="", flush=True)

    async for chunk in client.chat("gpt-4o-mini").stream_async("Write a haiku about timber."):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()
    asyncio.run(main_async())
