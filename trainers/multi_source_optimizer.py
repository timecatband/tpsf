import torch
import torchaudio

# Encapsulate the training loop for optimizing the mix of two
# audio files by tweaking a parametric effect pipeline for each one
# (and with respect to an objective). Use a single optimizer for all!

class MultiSourceOptimizer:
    def __init__(self, audio_files, effect_pipelines, objective):
        self.audio_files = audio_files
        self.effect_pipelines = effect_pipelines
        self.objective = objective
    
    def optimize(self, num_steps=1000, learning_rate=0.01):
        # Is this discarding stereo?
        audio_data = []
        sr = None
        for audio_file in self.audio_files:
            waveform, sr = torchaudio.load(audio_file)
            audio_data.append(waveform)
            if sr is None:
                sr = sr
            elif sr != sr:
                raise ValueError("Mismatched sample rates")
        optim = torch.optim.Adam([effect_pipeline.parameters() for effect_pipeline in self.effect_pipelines], lr=learning_rate)
        for i in range(num_steps):
            optim.zero_grad()
            processed_waveforms = [effect_pipeline(waveform) for waveform, effect_pipeline in zip(audio_data, self.effect_pipelines)]
            # Mix processed waveforms
            mixed_waveform = sum(processed_waveforms)*(1/len(processed_waveforms))
            loss = self.objective(*processed_waveforms, mixed_waveform)
            loss.backward()
            optim.step()
        return mixed_waveform
        