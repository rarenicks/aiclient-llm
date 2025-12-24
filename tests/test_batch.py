"""
Tests for helper batch processing.
"""
import pytest
import asyncio
import time
from aiclient.batch import BatchProcessor

@pytest.mark.asyncio
async def test_batch_processor_concurrency():
    """Test that concurrency limit is respected."""
    # We will run 10 tasks that take 0.1s each.
    # With concurrency 1, it should take ~1.0s
    # With concurrency 5, it should take ~0.2s
    
    async def fast_task(x):
        await asyncio.sleep(0.1)
        return x * 2
        
    inputs = list(range(10))
    processor = BatchProcessor(concurrency=5)
    
    start = time.time()
    results = await processor.process(inputs, fast_task)
    end = time.time()
    
    duration = end - start
    assert 0.15 < duration < 0.6  # approx 0.2s + overhead
    assert len(results) == 10
    assert results == [x * 2 for x in inputs]

@pytest.mark.asyncio
async def test_batch_processor_exceptions():
    """Test exception handling."""
    async def faulty_task(x):
        if x == 3:
            raise ValueError("Boom")
        return x
        
    inputs = [1, 2, 3, 4]
    processor = BatchProcessor(concurrency=2)
    
    results = await processor.process(inputs, faulty_task, return_exceptions=True)
    
    assert results[0] == 1
    assert results[1] == 2
    assert isinstance(results[2], ValueError)
    assert str(results[2]) == "Boom"
    assert results[3] == 4

@pytest.mark.asyncio
async def test_batch_processor_raise_exception():
    """Test exception raising mode."""
    async def faulty_task(x):
        if x == 1:
            raise ValueError("Boom")
        return x
        
    inputs = [1]
    processor = BatchProcessor()
    
    with pytest.raises(ValueError, match="Boom"):
        await processor.process(inputs, faulty_task, return_exceptions=False)
