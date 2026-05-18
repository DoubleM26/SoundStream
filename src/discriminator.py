from torch import nn
import torchaudio
import torch


class WaveDiscriminator(nn.Module):
    """
    15 x 1, stride=1 conv, channels=16, LeakyReLU
    41 x 1, stride=4, groups=4 conv, channels=64, LeakyReLU
    41 x 1, stride=4, groups=16 conv, channels=256, LeakyReLU
    41 x 1, stride=4, groups=64 conv, channels=1024, LeakyReLU
    41 x 1, stride=4, groups=256 conv, channels=1024, LeakyReLU
    5 x 1, stride=1 conv, channels=1024, LeakyReLU
    3 x 1, stride=1 conv, channels=1
    """

    # layer norm ?
    def __init__(self, scale):
        super().__init__()
        self.scale_block = nn.Identity()
        if scale == 2:
            self.scale_block = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        if scale == 4:
            self.scale_block = nn.Sequential(
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
            )
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(kernel_size=15, stride=1, in_channels=1, out_channels=16, padding=7),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=4, in_channels=16, out_channels=64, padding=20),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=16, in_channels=64, out_channels=256, padding=20),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=64, in_channels=256, out_channels=1024, padding=20),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=256, in_channels=1024, out_channels=1024, padding=20),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=5, stride=1, groups=1, in_channels=1024, out_channels=1024, padding=2),
                nn.LeakyReLU(0.2),
            ),
            nn.Conv1d(kernel_size=3, stride=1, groups=1, in_channels=1024, out_channels=1, padding=1),
        ])

    def forward(self, x):
        feature_map = []
        x = self.scale_block(x)
        for block in self.net:
            x = block(x)
            feature_map.append(x)
        return feature_map


class ResidualUnit(nn.Module):
    def __init__(self, n, m, s):
        super().__init__()
        self.s = s

        self.net = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=n, out_channels=n, padding="same"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(kernel_size=(s[0] + 2, s[1] + 2), stride=s, in_channels=n, out_channels=m * n, padding=(1, 1)),
        )

        self.skip = nn.Conv2d(kernel_size=(1, 1), stride=s, in_channels=n, out_channels=m * n)
        self.final_activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.final_activation(self.net(x) + self.skip(x))


class STFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        C = 32
        self.blocks = nn.ModuleList([
            nn.Conv2d(kernel_size=(7, 7), in_channels=2, out_channels=C, padding=3),
            ResidualUnit(C, 2, (1, 2)),
            ResidualUnit(2 * C, 2, (2, 2)),
            ResidualUnit(4 * C, 1, (1, 2)),
            ResidualUnit(4 * C, 2, (2, 2)),
            ResidualUnit(8 * C, 1, (1, 2)),
            ResidualUnit(8 * C, 2, (2, 2)),
            nn.Conv2d(kernel_size=(1, 8), in_channels=C * 16, out_channels=1),
        ])

        self.STFT = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=None,
        )  # center?

    def waveform_to_stft(self, waveform):
        waveform = waveform.squeeze(1)
        stft_waveform = self.STFT(waveform)
        stft_waveform = stft_waveform[:, :-1, :]
        stft = torch.view_as_real(stft_waveform)
        stft = stft.permute(0, 3, 2, 1)

        return stft

    def forward(self, x):
        feature_map = []
        x = self.waveform_to_stft(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            feature_map.append(x)
        return feature_map


class FullDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            WaveDiscriminator(1),
            WaveDiscriminator(2),
            WaveDiscriminator(4),
            STFTDiscriminator()
        ])

    def forward(self, x):
        res = []
        for d in self.discriminators:
            res.append(d(x))

        return res
