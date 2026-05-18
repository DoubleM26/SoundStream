from torch import nn
import torch.nn.functional as F
from src.model.components import ResidualUnit, CausalConv, CausalConvT


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            CausalConvT(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride
            ),
            ResidualUnit(out_channels, out_channels, 1),
            ResidualUnit(out_channels, out_channels, 3),
            ResidualUnit(out_channels, out_channels, 9)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, c, k, strides):
        super().__init__()
        c *= 2 ** len(strides)

        layers = [CausalConv(in_channels=k, out_channels=c, kernel_size=7)]
        for stride in reversed(strides):
            layers.append(DecoderBlock(c, c // 2, stride))
            c //= 2
        layers.append(nn.ELU())
        layers.append(CausalConv(in_channels=c, out_channels=1, kernel_size=7))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
