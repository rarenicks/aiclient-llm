import asyncio
import json
from typing import Any, Dict, Iterator, List, Type, TypeVar, Union

from pydantic import BaseModel

from ..data_types import BaseMessage, ModelResponse, UserMessage
from ..middleware import Middleware
from ..providers.base import Provider
from ..transport.base import Transport
from ..utils import should_retry

T = TypeVar("T", bound=BaseModel)


class ChatModel:
    """Wrapper for chat model interactions using a Provider strategy."""

    def __init__(
        self,
        model_name: str,
        provider: Provider,
        transport: Transport,
        middlewares: List[Middleware] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.provider = provider
        self.transport = transport
        self.middlewares = middlewares or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: Union[str, List[BaseMessage]],
        response_model: Type[T] = None,
        strict: bool = False,
        tools: List[Any] = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        top_k: int = None,
        stop: Union[str, List[str]] = None,
    ) -> Union[ModelResponse, T]:
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
        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            result = mw.before_request(self.model_name, messages)
            if isinstance(result, ModelResponse):
                # Short-circuit: return cached/mocked response immediately
                return result
            messages = result

        # 3. Handling Structured Output
        response_schema = None
        if response_model:
            response_schema = response_model.model_json_schema()

            # If NOT strict, fallback to legacy prompt injection
            if not strict:
                instruction = (
                    "\n\nRestricted Output Mode: You must response strictly with a "
                    "valid JSON object that matches the following JSON Schema.\n"
                    "Do not return the schema itself. Return the data instance.\n"
                    f"Schema:\n{json.dumps(response_schema, indent=2)}"
                )
                if messages and isinstance(messages[-1], UserMessage):
                    last_msg = messages[-1]
                    new_content = last_msg.content + instruction
                    messages[-1] = UserMessage(content=new_content)
                else:
                    messages.append(UserMessage(content=instruction))

        # 4. Execute Request
        endpoint, data = self.provider.prepare_request(
            self.model_name,
            messages,
            tools=tools,
            response_schema=response_schema if strict else None,
            strict=strict,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        response_data = None
        for attempt in range(self.max_retries + 1):
            try:
                response_data = self.transport.send(endpoint, data)
                break
            except Exception as e:
                # Notify middleware of error
                for mw in self.middlewares:
                    mw.on_error(e, self.model_name, attempt=attempt)

                # We assume transport raises exceptions that should_retry can inspect
                # Specifically HTTPTransport raises httpx.HTTPStatusError for 4xx/5xx if
                # raise_for_status() used or we might catch generic Exception and check
                # attributes.
                # `should_retry` handles it safely.
                if attempt == self.max_retries:
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
                raise ValueError(
                    f"Failed to parse structured output: {e}. "
                    f"Raw: {model_response.text}"
                )

        return model_response

    async def generate_async(
        self,
        prompt: Union[str, List[BaseMessage]],
        response_model: Type[T] = None,
        strict: bool = False,
        tools: List[Any] = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        top_k: int = None,
        stop: Union[str, List[str]] = None,
    ) -> Union[ModelResponse, T]:
        """
        Generate a response asynchronously.
        """
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]

        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            if hasattr(mw, "before_request_async") and asyncio.iscoroutinefunction(mw.before_request_async):
                result = await mw.before_request_async(self.model_name, messages)
            else:
                 result = mw.before_request(self.model_name, messages)
            
            if isinstance(result, ModelResponse):
                return result
            messages = result

        # 3. Handling Structured Output
        response_schema = None
        if response_model:
            response_schema = response_model.model_json_schema()

            if not strict:
                instruction = (
                    "\n\nRestricted Output Mode: You must response strictly with a "
                    "valid JSON object that matches the following JSON Schema.\n"
                    "Do not return the schema itself. Return the data instance.\n"
                    f"Schema:\n{json.dumps(response_schema, indent=2)}"
                )
                if messages and isinstance(messages[-1], UserMessage):
                    last_msg = messages[-1]
                    new_content = last_msg.content + instruction
                    messages[-1] = UserMessage(content=new_content)
                else:
                    messages.append(UserMessage(content=instruction))

        # 4. Execute Request
        endpoint, data = self.provider.prepare_request(
            self.model_name,
            messages,
            tools=tools,
            response_schema=response_schema if strict else None,
            strict=strict,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        response_data = None
        for attempt in range(self.max_retries + 1):
            try:
                response_data = await self.transport.send_async(endpoint, data)
                break
            except Exception as e:
                # Notify middleware of error
                for mw in self.middlewares:
                    if hasattr(mw, "on_error_async"):
                        await mw.on_error_async(e, self.model_name, attempt=attempt)
                    elif asyncio.iscoroutinefunction(mw.on_error):
                        await mw.on_error(e, self.model_name, attempt=attempt)
                    else:
                        mw.on_error(e, self.model_name, attempt=attempt)

                if attempt == self.max_retries or not should_retry(e):
                    raise e
                else:
                    wait_time = self.retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)

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
                raise ValueError(
                    f"Failed to parse structured output: {e}. "
                    f"Raw: {model_response.text}"
                )

        return model_response

    async def stream_async(
        self,
        prompt: Union[str, List[BaseMessage]],
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stop: Union[str, List[str]] = None,
    ) -> Iterator[str]:
        """Stream a response asynchronously."""
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]

        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            if hasattr(mw, "before_request_async") and asyncio.iscoroutinefunction(mw.before_request_async):
                messages = await mw.before_request_async(self.model_name, messages)
            else:
                messages = mw.before_request(self.model_name, messages)

        # 3. Execute Request
        endpoint, data = self.provider.prepare_request(
            self.model_name,
            messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        try:
            async for chunk_data in self.transport.stream_async(endpoint, data):
                chunk = self.provider.parse_stream_chunk(chunk_data)
                if chunk:
                    yield chunk.text
        except Exception as e:
            for mw in self.middlewares:
                mw.on_error(e, self.model_name)
            raise e

    def stream(
        self,
        prompt: Union[str, List[BaseMessage]],
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stop: Union[str, List[str]] = None,
    ) -> Iterator[str]:
        """Stream a response synchronously."""
        # 1. Prepare Messages
        messages = prompt
        if isinstance(prompt, str):
            messages = [UserMessage(content=prompt)]

        # 2. Middleware Hook: before_request
        for mw in self.middlewares:
            messages = mw.before_request(self.model_name, messages)

        # 3. Execute Request
        endpoint, data = self.provider.prepare_request(
            self.model_name,
            messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        try:
            for chunk_data in self.transport.stream(endpoint, data):
                chunk = self.provider.parse_stream_chunk(chunk_data)
                if chunk:
                    yield chunk.text
        except Exception as e:
            for mw in self.middlewares:
                mw.on_error(e, self.model_name)
            raise e


class SimpleResponse:
    def __init__(self, text: str, raw: Dict[str, Any]):
        self.text = text
        self.raw = raw
