from torch import nn
from src.model.components import ResidualUnit, CausalConv


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1),
            ResidualUnit(in_channels, in_channels, 3),
            ResidualUnit(in_channels, in_channels, 9),
            CausalConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride
            )
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, c, k, strides):
        super().__init__()
        layers = [CausalConv(in_channels=1, out_channels=c, kernel_size=7)]
        for stride in strides:
            layers.append(EncoderBlock(c, c * 2, stride))
            c *= 2
        layers.append(nn.ELU())
        layers.append(CausalConv(in_channels=c, out_channels=k, kernel_size=3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

