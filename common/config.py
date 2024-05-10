import json
import torch
import torchaudio
import numpy as np
from synth.oscillator import HarmonicEmbedder


def stretch_signal(signal, stretched_size):
    stretched_samples_per_original_sample = stretched_size / signal.shape[0]
    stretched_components = []
    for i in range(signal.shape[0]):
        # Repeat signal i by stretched_samples_per_original_sample times
        stretched_sample = signal[i].repeat(int(stretched_samples_per_original_sample))
        # Mix in the next sample
        if i + 1 < signal.shape[0]:
            blend = torch.linspace(1, 1/stretched_samples_per_original_sample, int(stretched_samples_per_original_sample))
            stretched_sample = stretched_sample * blend + signal[i + 1] * (1 - blend)
        stretched_components.append(stretched_sample)
    stretched_signal = torch.cat(stretched_components)
    return stretched_signal

def parse_synth_experiment_config_from_file(experiment_dir):
    defaults={}
    defaults["time_latent_size"] = 2
    defaults["latent_size"] = 8
    
    
    config = None
    
    file_path = experiment_dir + "/experiment.json"
    with open(file_path) as f:
        config = json.load(f)
    parsed_config = {}
    # Merge defaults with config
    for key in defaults:
        if key not in config:
            config[key] = defaults[key]
    parsed_config["latent_size"] = config["latent_size"]
    parsed_config["time_latent_size"] = config["time_latent_size"]
    parsed_config["harmonic_embedder"] = HarmonicEmbedder(config["latent_size"], config["time_latent_size"])
    
    parsed_config["midi_file"] = experiment_dir + "/" + config["midi_file"]
    parsed_config["wav_file"] = config["wav_file"]
    effect_chain_string = ""
    effects = config["effect_chain"]
    for effect in effects:
        effect_chain_string += effect + ","
    parsed_config["effect_chain"] = effect_chain_string
    synth_string = ""
    synths = config["synths"]
    for synth in synths:
        synth_string += synth + ","
    parsed_config["synths"] = synth_string
    
    target_audio_path = experiment_dir + "/" + config["wav_file"]
    target_audio, sr = torchaudio.load(target_audio_path)
    parsed_config["target_audio"] = target_audio
    parsed_config["sr"] = sr

    if "loudness" in config:
        loudness_npy = config["loudness"]
        pitch_npy = config["pitch"]
        loudness_npy = experiment_dir + "/" + loudness_npy
        pitch_npy = experiment_dir + "/" + pitch_npy
        loudness_npy = np.load(loudness_npy)
        pitch_npy = np.load(pitch_npy)
        loudness = torch.tensor(loudness_npy)
        pitch = torch.tensor(pitch_npy)
        loudness = stretch_signal(loudness, target_audio.shape[1])
        pitch = stretch_signal(pitch, target_audio.shape[1])
        parsed_config["loudness"] = loudness
        parsed_config["pitch"] = pitch
    weights = None
    if "weights" in config:
        weights = config["weights"]
        weights = file_path + "/" + weights
    parsed_config["weights"] = weights
    return parsed_config
    