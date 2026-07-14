"""
OMR Plugins Module — Registration and loading for third-party extensions.
"""
from typing import Dict, Type, Any

_PLUGINS: Dict[str, Type] = {}


def register_plugin(name: str):
    """
    Decorator to register a third-party plugin class.
    
    Example:
        @register_plugin("medical")
        class MedicalPlugin:
            def validate(self, dataset): ...
    """
    def wrapper(cls):
        _PLUGINS[name] = cls
        return cls
    return wrapper


def get_plugin(name: str) -> Any:
    """Instantiates and returns a registered plugin by name."""
    if name not in _PLUGINS:
        raise ValueError(f"Plugin '{name}' not found. Did you pip install it?")
    return _PLUGINS[name]()
