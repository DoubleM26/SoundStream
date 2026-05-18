from torch import nn


def l1_loss(audio_in, audio_out):
    loss = nn.L1Loss()
    return loss(audio_in, audio_out)
