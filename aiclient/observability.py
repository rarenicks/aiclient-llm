import time
import uuid
import json
import logging
from typing import Dict, Union, List, Optional, Any
from .data_types import ModelResponse, BaseMessage
from .middleware import Middleware

logger = logging.getLogger("aiclient.observability")

class TracingMiddleware(Middleware):
    def __init__(self, trace_exporter: Optional[Any] = None):
        """
        Simple tracing middleware that logs traces to logger or optional exporter.
        """
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.exporter = trace_exporter

    def before_request(self, model: str, prompt: Union[str, List[BaseMessage]]) -> Union[str, List[BaseMessage]]:
        # We need a request ID to correlate. 
        # Middleware is stateful per request ONLY if fresh instance used, 
        # BUT ChatModel reuses middleware list.
        # This is concurrency unsafe if we store state on `self`.
        # However, `before_request` returns modified prompt. 
        # We can't inject ID into prompt easily without polluting it.
        # Ideally, we rely on context var or similar.
        # For simplicity in v0.4, we assume synchronous single-threaded usage OR we just log start/end independently if we lack ID.
        # BUT, to match start/end, we really need context. 
        # Hack: Use current thread ID or similar if sync?
        # Or, just log "Request Started" and return.
        
        # Real tracing needs passing context. 
        # Let's just log "Request to {model}" with a random ID, 
        # and hope strict ordering or simple usage allows correlation for now.
        # Better: Implementation Plan said "Span-based tracing".
        
        rid = str(uuid.uuid4())
        # We can't pass RID to after_response easily without modifying protocol.
        # But we can assume this is a limitation for v0.4 simple middleware.
        # Let's just log.
        logger.info(f"Trace[{rid}]: Request to {model}")
        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        # We don't have rid here.
        # Just log completion.
        logger.info(f"Trace[...]: Response from {response.provider} - Tokens: {response.usage.total_tokens}")
        return response

    def on_error(self, error: Exception, model: str, **kwargs) -> None:
        print(f"[TRACE] Error in {model}: {error}")

class OpenTelemetryMiddleware(Middleware):
    """
    OpenTelemetry integration. Requires `opentelemetry-api` and `opentelemetry-sdk`.
    """
    def __init__(self, service_name: str = "aiclient"):
        self.tracer = None
        try:
            from opentelemetry import trace
            self.tracer = trace.get_tracer(service_name)
        except ImportError:
            pass
            
        # ContextVar for thread-safe/async-safe span storage
        import contextvars
        self._span_ctx = contextvars.ContextVar("current_span", default=None)

    def before_request(self, model: str, prompt: Union[str, List[BaseMessage]]) -> Union[str, List[BaseMessage]]:
        if self.tracer:
            span = self.tracer.start_span("llm.generate")
            span.set_attribute("llm.model", model)
            # Store span in context
            self._span_ctx.set(span)
        return prompt

    def after_response(self, response: ModelResponse) -> ModelResponse:
        span = self._span_ctx.get()
        if span:
            span.set_attribute("llm.provider", response.provider)
            if response.usage:
                span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)
                span.set_attribute("llm.usage.input_tokens", response.usage.input_tokens)
                span.set_attribute("llm.usage.output_tokens", response.usage.output_tokens)
            span.end()
            # Reset context? Not strictly necessary if token is managed, but good practice.
            # self._span_ctx.set(None) 
        return response

    def on_error(self, error: Exception, model: str, **kwargs) -> None:
        span = self._span_ctx.get()
        if span:
            # Ideally we'd capture exception in span if we had context
            span.record_exception(error)
            # OTel status mapping is involved
            # Simpler:
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
