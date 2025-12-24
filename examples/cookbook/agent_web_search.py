"""
Cookbook: Agent with Web Search 🕵️‍♂️
Run: python examples/cookbook/agent_web_search.py

This example demonstrates how to build an Agent capable of "searching the web".
We mock the search tool for simplicity, but you can swap it with `duckduckgo-search`.
"""
import os
import json
from dotenv import load_dotenv
from aiclient import Client, Agent

load_dotenv()

# --- Tools ---
def search_web(query: str) -> str:
    """
    Simulates a web search engine. 
    Returns a JSON string summary of results.
    """
    print(f"\n[Tool] Searching web for: {query!r}")
    
    # Mock results
    mock_results = {
        "python 3.14 release date": "Python 3.14 is expected to be released in October 2025.",
        "current weather in tokyo": "It is currently 18°C and cloudy in Tokyo.",
        "who won the super bowl 2024": "The Kansas City Chiefs won Super Bowl LVIII."
    }
    
    # Fuzzy match for demo
    key = next((k for k in mock_results if k in query.lower()), None)
    if key:
        return mock_results[key]
    
    return "No relevant results found."

def get_current_time() -> str:
    """Returns the current server time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Main ---
async def main_async():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env")
        return

    client = Client()
    
    # Initialize Agent with tools
    agent = Agent(
        model=client.chat("gpt-4o"),
        tools=[search_web, get_current_time],
        max_steps=5
    )
    
    print("🕵️‍♂️ Agent Ready. Tools: search_web, get_current_time")
    
    questions = [
        "What time is it right now?",
        "When is Python 3.14 coming out?",
    ]
    
    # Use async context manager if agent needs it (e.g. for MCP), 
    # though for local tools it might not be strictly necessary, it's good practice.
    async with agent:
        for q in questions:
            print(f"\nUser: {q}")
            response = await agent.run_async(q)
            print(f"Agent: {response}\n" + "-"*40)

def main():
    import asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
