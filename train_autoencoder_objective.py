import torch
import torch.nn as nn
import sys
import torchaudio
from trainers.train_autoencoder import EqFeatureAutoencoderTrainer

audio_file, sr = torchaudio.load(sys.argv[1])
audio_file = audio_file[0].unsqueeze(0)
autoencoder = EqFeatureAutoencoderTrainer(256)
autoencoder.train(audio_file)

output_model_file = sys.argv[2]
torch.save(autoencoder.model.state_dict(), output_model_file)