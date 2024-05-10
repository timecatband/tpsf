import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math
import torchaudio
import sys

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


def extract_pitch(signal, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0

audio_file = sys.argv[1]
target_audio, sr = torchaudio.load(audio_file)
target_audio = target_audio.numpy()
block_size = 256
loudness = extract_loudness(target_audio[0], sr, block_size)
pitch = extract_pitch(target_audio[0], sr, block_size)

experiment_dir = sys.argv[2]

import os
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

np.save(experiment_dir + "/loudness.npy", loudness)
np.save(experiment_dir + "/pitch.npy", pitch)

