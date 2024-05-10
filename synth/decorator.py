from typing import get_type_hints

class synthd:
    def __init__(self, name: str):
        self.name = name

def synthd(name: str):
    def decorator(cls):
        cls.__synth_name__ = name  # Store effect name in the class
        return cls
    return decorator
