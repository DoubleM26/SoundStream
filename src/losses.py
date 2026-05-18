from torch import nn
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
import math


def l1_loss(audio_in, audio_out):
    loss = nn.L1Loss()
    return loss(audio_in, audio_out)


def discriminator_loss(real, fake, config):
    res = {
        "loss_real": 0,
        "loss_fake": 0
    }
    for r in real:
        logits = r[-1]
        res["loss_real"] += torch.relu(1 - logits).mean()

    for f in fake:
        logits = f[-1]
        res["loss_fake"] += torch.relu(1 + logits).mean()

    total_loss = (res["loss_real"] + res["loss_fake"]) / len(real)
    res["total_loss"] = total_loss

    return total_loss, res


def adversarial_loss(fake):  # hinge
    loss = 0
    for f in fake:
        logits = f[-1]
        loss += torch.relu(1.0 - logits).mean()
    return loss / len(fake)


def feature_matching_loss(real, fake):
    loss = 0
    cnt = 0
    for i in range(len(real)):
        for real_feature, fake_feature in zip(real[i][:-1], fake[i][:-1]):
            loss += F.l1_loss(fake_feature, real_feature.detach())
            cnt += 1
    return loss / cnt


class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scales = config["reconstruction_loss"]["scales"]
        mels_list = []
        for scale in self.scales:
            mels_list.append(
                MelSpectrogram(
                    sample_rate=config["data"]["sample_rate"],
                    n_fft=scale,
                    hop_length=scale // 4,
                    n_mels=config["reconstruction_loss"]["n_mels"],
                )  # power?
            )
        self.mels = nn.ModuleList(mels_list)

    def forward(self, real, fake):
        real = real.squeeze(1)
        fake = fake.squeeze(1)

        loss = 0

        for scale, mel in zip(self.scales, self.mels):
            real_mel = mel(real)
            fake_mel = mel(fake)

            l1 = torch.mean(torch.abs(real_mel - fake_mel))  # sum or mean check
            l2 = torch.sqrt(torch.mean((torch.log(real_mel + 1e-6) - torch.log(fake_mel + 1e-6)) ** 2))

            loss += l1 + math.sqrt(scale / 2) * l2
        return loss


def generator_loss(real_wf, fake_wf, real, fake, rec_loss, config):
    # TODO: commitment los
    res = {
        "reconstruction_loss": rec_loss(real_wf, fake_wf),
        "adversarial_loss": adversarial_loss(fake),
        "feature_matching_loss": feature_matching_loss(real, fake),
        "commitment_loss": 0
    }

    w = config["generator_loss_weights"]

    total_loss = (w["reconstruction"] * res["reconstruction_loss"] +
                  w["adversarial"] * res["adversarial_loss"] +
                  w["feature_matching"] * res["feature_matching_loss"] +
                  w["commitment"] * res["commitment_loss"]
                  )
    res["total_loss"] = total_loss
    return total_loss, res
