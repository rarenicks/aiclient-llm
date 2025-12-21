from typing import Callable, Type, Any, Dict
from pydantic import BaseModel

class Tool:
    """
    A definition for a tool that can be used by an AI model.
    wraps a function and its Pydantic schema.
    """
    def __init__(self, name: str, fn: Callable, schema: Type[BaseModel], description: str = ""):
        self.name = name
        self.fn = fn
        self.args_schema = schema
        self.description = description or fn.__doc__ or ""

    @property
    def schema(self) -> Dict[str, Any]:
        """JSON Schema for the tool arguments."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }

    @classmethod
    def from_fn(cls, fn: Callable) -> "Tool":
        import inspect
        from pydantic import create_model, Field
        
        sig = inspect.signature(fn)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self": continue
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                annotation = str
            
            default = param.default
            if default == inspect.Parameter.empty:
                 params[name] = (annotation, ...)
            else:
                 params[name] = (annotation, default)
                 
        schema = create_model(f"{fn.__name__}Schema", **params)
        return cls(name=fn.__name__, fn=fn, schema=schema)

    def run(self, **kwargs) -> Any:
        # Validate arguments using schema
        args = self.args_schema(**kwargs)
        return self.fn(**args.model_dump())
