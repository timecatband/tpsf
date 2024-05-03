from typing import get_type_hints

class Effect:
    def __init__(self, name: str):
        self.name = name

def effect(name: str):
    def decorator(cls):
        cls.__effect_name__ = name  # Store effect name in the class
        return cls
    return decorator
