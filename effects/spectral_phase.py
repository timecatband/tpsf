import torch
import torch.nn as nn
import numpy as np
class PhaseDistortionField(nn.Module):
  def __init__(self):
    super(PhaseDistortionField, self).__init__()
    size = 64
    self.layers = nn.Sequential(
      nn.Linear(2, size),
      nn.ReLU(),
      nn.Linear(size, 1),
      nn.ReLU())
    self.layers2 = nn.Sequential(
        nn.Linear(2, size),
        nn.ReLU(),
        nn.Linear(size, size),
        nn.ReLU(),
        nn.Linear(size,size),
        nn.ReLU(),
        nn.Linear(size, 1))
    
  def forward(self, x, y, phase):

      x = x.flatten()
      y = y.flatten()
      phase = phase.flatten()
      x = x.unsqueeze(-1)
      y = y.unsqueeze(-1)
      x = x.float()
      y = y.float()
   #   x = x / torch.max(x)
   #   y = y / torch.max(y)

      
      phase = phase.unsqueeze(-1)
      input = torch.cat([x, y], dim=-1)
      input = input.float()
      remapped = self.layers(input)*np.pi*2
      # Wrap the phase to [-pi, pi]
      #remapped = torch.remainder(remapped, 2 * np.pi)
      return phase + remapped
  
  
def spectrogram_phase_modify_audio(audio_tensor, audio_sample_rate, phase_transform, samples_per_network_batch=16384):
    # STFT Parameters (adjust as needed)
    n_fft = 2048
    hop_length = 128
    window = torch.hann_window(n_fft).to(audio_tensor.device)
    # Cast to float32
    audio_tensor = audio_tensor.to(torch.float32)

    # Compute complex spectrogram using STFT
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    print("spectrogram shape", spectrogram.shape)
    print("spectrogram dtype", spectrogram.dtype)
    print("original audio shape", audio_tensor.shape)

    magnitudes = torch.abs(spectrogram[0])
    phases = spectrogram[1].unsqueeze(0)  # Unsqueeze to match the shape of the magnitude spectrogram
    phases = torch.angle(phases)    
    phases = torch.unsqueeze(phases, -1)

    # Vectorized Phase Transformation (single batch)
    x_coords, y_coords = torch.meshgrid(torch.arange(spectrogram.shape[1]),
                                        torch.arange(spectrogram.shape[2]))
    all_phases = phases[0, x_coords, y_coords]
    
    # Apply phase transform batch by batch ensuring we process all samples
    transformed_phases = []
    spectrogram_time_samples_per_batch = hop_length * samples_per_network_batch // audio_sample_rate
    samples_per_network_batch = spectrogram_time_samples_per_batch
    # TODO: This probably skips some at the end
    for i in range(0, all_phases.shape[0], samples_per_network_batch):
        phases_batch = all_phases[i:i + samples_per_network_batch]
        x_coords_batch = x_coords[i:i + samples_per_network_batch]/x_coords.max()
        y_coords_batch = y_coords[i:i + samples_per_network_batch]/y_coords.max()
        transformed_phases_batch = phase_transform(x_coords_batch, y_coords_batch, phases_batch)
        transformed_phases.append(transformed_phases_batch)
    transformed_phases = torch.cat(transformed_phases, dim=0)
    
    # Apply vectorized phase transform
#transformed_phases = phase_transform(x_coords, y_coords, all_phases, x_offset)
    # Restore original shape of phases
  #  transformed_phases = transformed_phases.reshape(phases.shape)
  #  print(transformed_phases.shape)
   # transformed_phases = torch.squeeze(transformed_phases, -1)
    # Resynthesize audio using ISTFT
    transformed_phases = transformed_phases.reshape(phases.shape)
    print("all phases and transformed phases shape", all_phases.shape, transformed_phases.shape)
    transformed_phases = torch.squeeze(transformed_phases, -1)
    modified_spectrogram = magnitudes * torch.exp(1j * transformed_phases)
    reconstructed_audio = torch.istft(modified_spectrogram, n_fft=n_fft, hop_length=hop_length, window=window)

    return reconstructed_audio

class PhaseDistortionFieldEffect(nn.Module):
    def __init__(self):
        super(PhaseDistortionFieldEffect, self).__init__()
        self.phase_distortion_field = PhaseDistortionField()

    def forward(self, x, sr):
        return spectrogram_phase_modify_audio(x, sr, self.phase_distortion_field)