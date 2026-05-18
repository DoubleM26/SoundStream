from torch import nn


class Simple(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3),
            nn.ELU(),
            nn.ConvTranspose1d(32, 1, 4, 2, 1)
        )

    def forward(self, x):
        return self.net(x)

