"""
Cookbook: Simple RAG Pipeline 🔍
Run: python examples/cookbook/rag_pipeline.py

This example demonstrates a basic Retrieval-Augmented Generation flow.
1. "Retrieve" relevant documents from a mock knowledge base.
2. Inject context into the prompt.
3. Answer the user's question based on context.
"""
import os
from typing import List
from dotenv import load_dotenv
from aiclient import Client
from aiclient.data_types import SystemMessage, UserMessage

load_dotenv()

# --- Mock Vector Database ---
KNOWLEDGE_BASE = [
    "The secret code is 'BLUEBERRY'.",
    "The project deadline is December 31st, 2025.",
    "To reset the device, hold the power button for 10 seconds.",
    "The CEO's favorite color is orange.",
    "Support hours are 9 AM to 5 PM EST."
]

def retrieve(query: str, k: int = 2) -> List[str]:
    """
    Naive retrieval: Find documents containing query keywords.
    In production, replace with vector similarity search (e.g., Chroma, Pinecone).
    """
    query_terms = set(query.lower().split())
    scores = []
    
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for term in query_terms if term in doc.lower())
        scores.append((score, doc))
        
    # Sort by score desc
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return top k with non-zero score
    return [doc for score, doc in scores if score > 0][:k]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env")
        return

    client = Client()
    model = client.chat("gpt-4o")
    
    print("📚 Knowledge Base Loaded.")
    
    while True:
        question = input("\nAsk a question: ")
        if question.lower() in ["quit", "exit"]:
            break
            
        # 1. Retrieve
        context_docs = retrieve(question)
        context_str = "\n".join(f"- {doc}" for doc in context_docs)
        
        if not context_docs:
            print("❌ No relevant info found in knowledge base.")
            continue
            
        print(f"🔍 Retrieved Context:\n{context_str}\n")
        
        # 2. Augment Prompt
        system_prompt = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the provided context below.\n\n"
            f"Context:\n{context_str}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=question)
        ]
        
        # 3. Generate
        print("AI: ", end="", flush=True)
        response = model.generate(messages)
        print(response.text)

if __name__ == "__main__":
    main()
