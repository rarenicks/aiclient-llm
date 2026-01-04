
import asyncio
import sys
import os
import random

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiclient.middleware import CostTrackingMiddleware
from aiclient.data_types import ModelResponse, Usage

async def run_simulated_request(tracker, model, delay, usage):
    """
    Simulates a request flow:
    1. before_request
    2. wait (simulating network io)
    3. after_response
    """
    # 1. Start
    tracker.before_request(model, "prompt")
    
    # 2. Wait (yield control to allow other tasks to interleave)
    await asyncio.sleep(delay)
    
    # 3. Finish
    response = ModelResponse(
        text="Response",
        raw={},
        usage=usage
    )
    tracker.after_response(response)

@pytest.mark.asyncio
async def test_concurrency_flaw_reproduction():
    """
    Simulates concurrent async tasks.
    """
    print("Starting reproduction test with asyncio tasks...")
    cost_tracker = CostTrackingMiddleware()
    
    # Request A: GPT-4 (Expensive)
    # Price: Input $30/1M. 1000 tokens = $0.03
    task_a = run_simulated_request(
        cost_tracker, 
        "gpt-4", 
        0.2, # Longer delay to ensure A finishes AFTER B starts
        Usage(input_tokens=1000, output_tokens=0, total_tokens=1000)
    )
    
    # Request B: GPT-3.5 (Cheap)
    # Price: Input $0.5/1M. 1000 tokens = $0.0005
    task_b = run_simulated_request(
        cost_tracker, 
        "gpt-3.5-turbo", 
        0.05, # Shorter delay, finishes first
        Usage(input_tokens=1000, output_tokens=0, total_tokens=1000)
    )
    
    # Run them concurrently
    await asyncio.gather(task_a, task_b)
    
    # Check results
    print(f"Total Cost Recorded: {cost_tracker.total_cost_usd:.6f}")
    
    # Expected total: 0.03 (A) + 0.0005 (B) = 0.0305
    # If bug exists (context clobbered by B): 0.0005 (A as B) + 0.0005 (B) = 0.0010
    
    # We verify it's close to 0.0305
    expected = 0.0305
    tolerance = 0.0001
    
    assert abs(cost_tracker.total_cost_usd - expected) < tolerance, \
        f"Race condition or calculation error! Expected ~{expected}, got {cost_tracker.total_cost_usd:.6f}"
    print("SUCCESS: Contexts were isolated. Correct cost calculated.")

if __name__ == "__main__":
    asyncio.run(test_concurrency_flaw_reproduction())
