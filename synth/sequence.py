import torch
import torch.nn as nn

import inspect
import torch.nn as nn
import synth 
from synth import *
from synth.oscillator import *
from synth.karplus import *
import re

def parse_synth_string(synth_string):
    pattern = r'([A-Za-z]+)(?:\[(.*?)\])?'
    return re.findall(pattern, synth_string)

def process_parameters(param_str):
    params = {}
    if param_str:
        for param in param_str.split(','):
            print("Parsing param", param)

            key, value = param.split('=')
            # Strip potential ] from param
            value = value.strip(']')
            
            try:
                params[key] = int(value)  # Try converting to int
            except ValueError:
                params[key] = float(value)  # Otherwise, assume float
    return params

def get_synth_classes(module):
    """Retrieves all classes within a module annotated with the 'synth' decorator."""
    synth_classes = {}
    print("Looking for synth classes in module", module)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, '__synth_name__'):
            synth_classes[obj.__synth_name__] = obj
            print(f"Found synth: {obj.__synth_name__}")
    return synth_classes

def build_synth_chain(synth_string):
    synth_components = parse_synth_string(synth_string)
    synth_classes = get_synth_classes(synth)

    modules = []
    for name, param_str in synth_components:
        if name in synth_classes:
            print("Building synth", name)
            kwargs = process_parameters(param_str)
            modules.append(synth_classes[name](**kwargs)) 
        else:
            raise ValueError(f"Unknown synth: {name}")

    return SynthSequence(*modules)

class SynthSequence(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    def forward(self, feedback_line, freq_rad: float, output_length_samples: int, latents, t, pitches=None):
        for module in self.children():
            feedback_line = module(feedback_line, freq_rad, output_length_samples, latents, t, pitches)
        return feedback_line