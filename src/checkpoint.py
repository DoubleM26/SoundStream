import os
from pathlib import Path

import torch


def save_checkpoint(path, model, discriminator, g_optimizer, d_optimizer, step, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "config": config,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path, model, discriminator, g_optimizer, d_optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    return checkpoint


def checkpoint_paths(config, step):
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    return (
        checkpoint_dir / "latest.pt",
        checkpoint_dir / f"step_{step:06d}.pt",
    )
