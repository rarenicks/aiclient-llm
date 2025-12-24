"""
Cookbook: Chatbot with Conversation Memory 🧠
Run: python examples/cookbook/chatbot_with_memory.py

This example demonstrates how to manage conversation history manually.
Features:
- Maintains a list of messages.
- Slide window truncation (rudimentary) to keep context within limits.
- System prompt for personality.
"""
import os
from typing import List
from dotenv import load_dotenv
from aiclient import Client
from aiclient.data_types import BaseMessage, UserMessage, SystemMessage, AssistantMessage

load_dotenv()

# Configuration
MAX_HISTORY = 10  # Keep last 10 messages
MODEL = "gpt-4o"

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env")
        return

    client = Client()
    
    # 1. Initialize Memory
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful and witty AI assistant.")
    ]
    
    print(f"🤖: Hello! I'm {MODEL}. Chat with me! (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # 2. Add User Message
            messages.append(UserMessage(content=user_input))
            
            # 3. Truncate History (Simple sliding window)
            # Keep System Prompt [0] + Last N messages
            if len(messages) > MAX_HISTORY + 1:
                # Determine how many to cut
                excess = len(messages) - (MAX_HISTORY + 1)
                # Slice: [System] + [Remaining...]
                # Note: This is naive. In production, count tokens!
                messages = [messages[0]] + messages[1+excess:]
            
            # 4. Generate Response
            print("AI: ", end="", flush=True)
            full_response = ""
            
            # Using streaming for better UX
            for chunk in client.chat(MODEL).stream(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            
            # 5. Add AI Message to History
            messages.append(AssistantMessage(content=full_response))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
