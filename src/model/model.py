import torch.nn.functional as F
import torch
from torch import nn
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.quantizer import ResidualVectorQuantizer


class SoundStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config["model"]["emb_dim"]
        strides = config["model"]["strides"]
        channels = config["model"]["channels"]
        self.encoder = Encoder(c=channels, k=emb_dim, strides=strides)
        self.quantizer = ResidualVectorQuantizer(config)
        self.decoder = Decoder(c=channels, k=emb_dim, strides=strides)

    def forward(self, x):
        length = x.shape[-1]
        z = self.encoder(x)
        q_out = self.quantizer(z)
        recon = self.decoder(q_out["quantized"])
        if recon.shape[-1] > length:
            recon = recon[..., :length]
        elif recon.shape[-1] < length:
            recon = F.pad(recon, (0, length - recon.shape[-1]))
        return {
            "recon": recon,
            "encoded": z,
            "commitment_loss": q_out["commitment_loss"],
            "rvq_stats": q_out["stats"],
        }
