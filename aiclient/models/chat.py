from typing import Iterator, Dict, Any, Union, List, Type, TypeVar, Optional
import json
from pydantic import BaseModel
from ..types import ModelResponse, BaseMessage, SystemMessage, UserMessage, AssistantMessage
from ..transport.base import Transport
from ..providers.base import Provider
from ..middleware import Middleware

T = TypeVar("T", bound=BaseModel)

import time
import asyncio
from ..utils import should_retry

class ChatModel:
    """Wrapper for chat model interactions using a Provider strategy."""
    def __init__(self, 
                 model_name: str, 
                 provider: Provider, 
                 transport: Transport, 
                 middlewares: List[Middleware] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.model_name = model_name
        self.provider = provider
        self.transport = transport
        self.middlewares = middlewares or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, prompt: Union[str, List[BaseMessage]], response_model: Type[T] = None, tools: List[Any] = None) -> Union[ModelResponse, T]:
        """
        Generate a response synchronously.
        If response_model is provided, returns an instance of that model.
        Otherwise returns ModelResponse.
        """
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]
        
        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            messages = mw.before_request(self.model_name, messages)

        # 3. Handling Structured Output (Inject after middleware to ensure it sticks, or before? 
        # Usually schema injection is "system" level, maybe typically middleware adds tracing/logging.
        # Let's keep schema injection logic here as "core" logic)
        if response_model:
            schema = response_model.model_json_schema()
            instruction = (
                f"\n\nRestricted Output Mode: You must response strictly with a valid JSON object that matches the following JSON Schema.\n"
                f"Do not return the schema itself. Return the data instance.\n"
                f"Schema:\n{json.dumps(schema, indent=2)}"
            )
            
            # Helper to append to list, careful not to mutate original input if it matters, 
            # but middleware might have returned new list
            # We copy to be safe? messages is typically a new list from middleware return
            if messages and isinstance(messages[-1], UserMessage):
                # Create new message to avoid mutating if shared ref
                last_msg = messages[-1]
                new_content = last_msg.content + instruction
                messages[-1] = UserMessage(content=new_content)
            else:
                 messages.append(UserMessage(content=instruction))

        # 4. Execute Request
        endpoint, data = self.provider.prepare_request(self.model_name, messages, tools=tools)
        
        response_data = None
        for attempt in range(self.max_retries + 1):
            try:
                response_data = self.transport.send(endpoint, data)
                break
            except Exception as e:
                # We assume transport raises exceptions that should_retry can inspect
                # Specifically HTTPTransport raises httpx.HTTPStatusError for 4xx/5xx if raise_for_status() used
                # or we might catch generic Exception and check attributes. `should_retry` handles it safely.
                if attempt < self.max_retries and should_retry(e):
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise e
        
        model_response = self.provider.parse_response(response_data)
        
        # 5. Middleware Hook: after_response
        for mw in self.middlewares:
            model_response = mw.after_response(model_response)
        
        # 6. Parse Structured Output
        if response_model:
            try:
                # Basic cleanup
                text = model_response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text.rsplit("\n", 1)[0]
                
                parsed = json.loads(text)
                return response_model.model_validate(parsed)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse structured output: {e}. Raw: {model_response.text}")

        return model_response

    async def generate_async(self, prompt: Union[str, List[BaseMessage]], response_model: Type[T] = None, tools: List[Any] = None) -> Union[ModelResponse, T]:
        """
        Generate a response asynchronously.
        """
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]
        
        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            messages = mw.before_request(self.model_name, messages)

        # 3. Handling Structured Output
        if response_model:
            schema = response_model.model_json_schema()
            instruction = (
                f"\n\nRestricted Output Mode: You must response strictly with a valid JSON object that matches the following JSON Schema.\n"
                f"Do not return the schema itself. Return the data instance.\n"
                f"Schema:\n{json.dumps(schema, indent=2)}"
            )
            if messages and isinstance(messages[-1], UserMessage):
                last_msg = messages[-1]
                new_content = last_msg.content + instruction
                messages[-1] = UserMessage(content=new_content)
            else:
                 messages.append(UserMessage(content=instruction))

        # 4. Execute Request
        endpoint, data = self.provider.prepare_request(self.model_name, messages, tools=tools)
        
        response_data = None
        for attempt in range(self.max_retries + 1):
            try:
                response_data = await self.transport.send_async(endpoint, data)
                break
            except Exception as e:
                if attempt < self.max_retries and should_retry(e):
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise e

        model_response = self.provider.parse_response(response_data)
        
        # 5. Middleware Hook: after_response
        for mw in self.middlewares:
            model_response = mw.after_response(model_response)
        
        # 6. Structured Output Parsing
        if response_model:
            try:
                text = model_response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text.rsplit("\n", 1)[0]
                parsed = json.loads(text)
                return response_model.model_validate(parsed)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse structured output: {e}. Raw: {model_response.text}")

        return model_response

    async def stream_async(self, prompt: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Stream a response asynchronously."""
         # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]

        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            messages = mw.before_request(self.model_name, messages)

        # 3. Execute Request
        endpoint, data = self.provider.prepare_request(self.model_name, messages, stream=True)
        async for chunk_data in self.transport.stream_async(endpoint, data):
            chunk = self.provider.parse_stream_chunk(chunk_data)
            if chunk:
                yield chunk.text

    def stream(self, prompt: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Stream a response synchronously."""
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]

        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            messages = mw.before_request(self.model_name, messages)

        # 3. Execute Request
        endpoint, data = self.provider.prepare_request(self.model_name, messages, stream=True)
        for chunk_data in self.transport.stream(endpoint, data):
            chunk = self.provider.parse_stream_chunk(chunk_data)
            if chunk:
                yield chunk.text

class SimpleResponse:
    def __init__(self, text: str, raw: Dict[str, Any]):
        self.text = text
        self.raw = raw
