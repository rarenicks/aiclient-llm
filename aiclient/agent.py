import asyncio
from typing import List, Any, Optional, Dict
from .models.chat import ChatModel
from .types import UserMessage, ToolMessage, AssistantMessage, BaseMessage
from .tools.base import Tool

class Agent:
    def __init__(self, model: ChatModel, tools: List[Any], max_steps: int = 10):
        self.model = model
        self.max_steps = max_steps
        # Normalize tools to aiclient.Tool objects if they aren't already
        self.tools = []
        self._tool_map: Dict[str, Tool] = {}
        
        for t in tools:
            if isinstance(t, Tool):
                tool_instance = t
            else:
                # Wrap callable
                tool_instance = Tool.from_fn(t)
            self.tools.append(tool_instance)
            self._tool_map[tool_instance.name] = tool_instance
            
        self.history: List[BaseMessage] = []

    async def run_async(self, prompt: str) -> str:
        """Asynchronous run loop."""
        self.history = [UserMessage(content=prompt)]
        
        for _ in range(self.max_steps):
            response = await self.model.generate_async(self.history, tools=self.tools)
            
            # Create AssistantMessage from response
            assistant_msg = AssistantMessage(
                content=response.text,
                tool_calls=response.tool_calls
            )
            self.history.append(assistant_msg)
            
            if not response.tool_calls:
                return response.text
                
            # Execute tools
            for tc in response.tool_calls:
                tool = self._tool_map.get(tc.name)
                if tool:
                    try:
                        # Assuming tool.run is sync/async? 
                        # Tool class currently wraps 'fn'. 
                        # We need to checking if fn is async or sync.
                        # For simplicity v0.3 let's assume sync tools or blocking run.
                        # Or check asyncio.iscoroutinefunction(tool.fn)
                        if asyncio.iscoroutinefunction(tool.fn):
                            result = await tool.fn(**tc.arguments)
                        else:
                            result = tool.fn(**tc.arguments)
                        
                        self.history.append(ToolMessage(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=str(result)
                        ))
                    except Exception as e:
                        self.history.append(ToolMessage(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=f"Error: {str(e)}"
                        ))
                else:
                    self.history.append(ToolMessage(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=f"Error: Tool {tc.name} not found"
                    ))
                    
        return "Max steps reached"

    def run(self, prompt: str) -> str:
        """Synchronous run loop wrapper."""
        return asyncio.run(self.run_async(prompt))

# ABORTING WRITE TO FIX TYPES.PY FIRST
