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
    
    def optimize(self, num_steps=1000, learning_rate=0.01, decay_lr = True):
        optimizer = torch.optim.AdamW(self.effect_pipeline.parameters(), lr=learning_rate)
        for i in range(num_steps):
            processed_waveform = self.effect_pipeline(self.audio_file.clone())#, self.sample_rate)
            loss = self.objective(processed_waveform)
            print("Step", i, "Loss", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Decay learning rate
            if decay_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.999
            if i % 100 == 0:
                torchaudio.save("output_intermediate.wav", processed_waveform.detach().cpu().unsqueeze(0), self.sample_rate)

        return processed_waveform
    
    