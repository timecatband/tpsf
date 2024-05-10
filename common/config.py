import json


def parse_synth_experiment_config_from_file(file_path):
    config = None
    with open(file_path) as f:
        config = json.load(f)
    