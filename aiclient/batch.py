import asyncio
from typing import List, Any, Callable, TypeVar, Coroutine, Union, Tuple
import logging

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger("aiclient.batch")

class BatchProcessor:
    """
    Helper to process async tasks in batch with concurrency limits.
    """
    def __init__(self, concurrency: int = 5):
        self.semaphore = asyncio.Semaphore(concurrency)

    async def process(
        self, 
        items: List[T], 
        func: Callable[[T], Coroutine[Any, Any, R]],
        return_exceptions: bool = True
    ) -> List[Union[R, Exception]]:
        """
        Process a list of items using the provided async function.
        
        Args:
            items: List of input data.
            func: Async function to call for each item.
            return_exceptions: If True, exceptions are returned as results instead of raising.
            
        Returns:
            List of results in the same order as items.
        """
        async def worker(item: T) -> Union[R, Exception]:
            async with self.semaphore:
                try:
                    return await func(item)
                except Exception as e:
                    if return_exceptions:
                        logger.error(f"Batch processing error for item {item}: {e}")
                        return e
                    raise

        tasks = [worker(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
