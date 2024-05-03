import torch
import torchaudio

# Encapsulate the training loop for optimizing a single audio file
# by tweaking a parametric effect pipeline (and with respect to an objective)
class SingleSourceOptimizer:
    def __init__(self, audio_file, sr, effect_pipeline, objective):
        self.audio_file = audio_file
        self.sample_rate = sr
        self.effect_pipeline = effect_pipeline
        self.objective = objective
        self.audio_file.requires_grad = False
    
    def optimize(self, num_steps=1000, learning_rate=0.01):
        optimizer = torch.optim.Adam(self.effect_pipeline.parameters(), lr=learning_rate)
        for i in range(num_steps):
            processed_waveform = self.effect_pipeline(self.audio_file)#, self.sample_rate)
            loss = self.objective(processed_waveform)
            print("Step", i, "Loss", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return processed_waveform
    
    