"""
Cookbook: Async Batch Processing ⚡
Run: python examples/cookbook/batch_processing.py

This example demonstrates how to process multiple prompts concurrently.
We use a MockProvider to simulate latency safely, but the same code works with real APIs.
"""
import asyncio
import time
from aiclient import Client
from aiclient.batch import BatchProcessor
from aiclient.testing import MockProvider, MockTransport  # Using mock for deterministic demo

async def slow_mock_generation(item: str):
    """
    Simulate a slow network call.
    In reality, this would be: 
    return await client.chat("model").generate_async(...)
    """
    # Simulate work
    await asyncio.sleep(0.5) 
    return f"Processed: {item}"

async def main():
    print("🚀 Starting Batch Processing Demo...")
    
    inputs = [f"Item {i}" for i in range(20)]
    
    # Process 20 items with concurrency limit of 5.
    # Theoretical time: (20 / 5) * 0.5s = 2.0s
    # Sequential time: 20 * 0.5s = 10.0s
    
    processor = BatchProcessor(concurrency=5)
    
    start_time = time.time()
    results = await processor.process(inputs, slow_mock_generation)
    duration = time.time() - start_time
    
    print(f"\n✅ Completed {len(results)} requests in {duration:.2f}s")
    print(f"Results preview: {results[:3]}...")
    
    # Verify strict ordering
    assert results[0] == "Processed: Item 0"
    assert results[-1] == "Processed: Item 19"

    # --- Real World Usage Example (Commented out) ---
    # client = Client()
    # async def call_llm(prompt):
    #     res = await client.chat("gpt-4o").generate_async(prompt)
    #     return res.text
    # results = await processor.process(my_prompts, call_llm)

if __name__ == "__main__":
    asyncio.run(main())
