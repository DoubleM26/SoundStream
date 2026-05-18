from torch import nn
import torch.nn.functional as F


class CausalConv(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dilation = kwargs.get('dilation', 1)
        kernel_size = kwargs.get('kernel_size', 1)
        self.pad = dilation * (kernel_size - 1)

    def forward(self, x):
        return super().forward(F.pad(x, [self.pad, 0]))


class CausalConvT(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size = kwargs.get("kernel_size", 1)
        stride = kwargs.get("stride", 1)
        dilation = kwargs.get("dilation", 1)

        self.pad = dilation * (kernel_size - 1) + 1 - stride

    def forward(self, x):
        x = super().forward(x)
        if self.pad > 0:
            x = x[..., :-self.pad]

        return x



class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation
            ),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.net(x)
