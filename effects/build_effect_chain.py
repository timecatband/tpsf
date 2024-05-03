import inspect
import torch.nn as nn
import effects 
from effects import *
import re

def parse_effect_string(effect_string):
    pattern = r'([A-Za-z]+)(?:\[(.*)\])?'  # Pattern to capture effect name and optional parameters
    return re.findall(pattern, effect_string)

def process_parameters(param_str):
    params = {}
    if param_str:
        for param in param_str.split(','):
            key, value = param.split('=')
            try:
                params[key] = int(value)  # Try converting to int
            except ValueError:
                params[key] = float(value)  # Otherwise, assume float
    return params

def get_effect_classes(module):
    """Retrieves all classes within a module annotated with the 'effect' decorator."""
    effect_classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, '__effect_name__'):
            effect_classes[obj.__effect_name__] = obj
            print(f"Found effect: {obj.__effect_name__}")
    return effect_classes

def build_effect_chain(effect_string):
    effect_components = parse_effect_string(effect_string)
    effect_classes = get_effect_classes(effects)

    modules = []
    for name, param_str in effect_components:
        if name in effect_classes:
            kwargs = process_parameters(param_str)
            modules.append(effect_classes[name](**kwargs)) 
        else:
            raise ValueError(f"Unknown effect: {name}")

    return nn.Sequential(*modules)