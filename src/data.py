import torch
import torchaudio
import torchcodec
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm


class SoundStreamDataset(Dataset):
    def __init__(self, root, sample_rate, crop_seconds):
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root)
        self.new_size = int(sample_rate * crop_seconds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        waveform, _, _, _, _, _ = self.ds[item]

        if waveform.size(1) < self.new_size:
            waveform = F.pad(waveform, (0, self.new_size - waveform.size(1)), mode="replicate")
        if waveform.size(1) > self.new_size:
            s = torch.randint(0, waveform.size(1) - self.new_size + 1, size=[1])[0].item()
            waveform = waveform[:, s:s + self.new_size]
        return waveform
