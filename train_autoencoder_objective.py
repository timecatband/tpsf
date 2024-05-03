import torch
import torch.nn as nn
import sys
import torchaudio
from trainers.train_autoencoder import EqFeatureAutoencoderTrainer

audio_file, sr = torchaudio.load(sys.argv[1])

autoencoder = EqFeatureAutoencoderTrainer(10)
autoencoder.train(audio_file)

output_model_file = sys.argv[2]
torch.save(autoencoder.model.state_dict(), output_model_file)