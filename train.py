import torch
from torch.utils.data import DataLoader
import comet_ml

from src.data import SoundStreamDataset
from src.discriminator import FullDiscriminator
from src.loops import train, initialize_codebook
from src.config import load_config
from src.losses import discriminator_loss, generator_loss, ReconstructionLoss
from src.model.model import SoundStream
from src.checkpoint import load_checkpoint
from src.validation import Validation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config_file',
                    type=str,
                    nargs='?',
                    default='./configs/debug.yml')
args = parser.parse_args()

config = load_config(args.config_file)

device = config['train']['device']

train_dataset = SoundStreamDataset("./", config["data"]["sample_rate"], config["data"]["crop_seconds"])
train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)

val = None
if config["validation"]["enabled"]:
    val = Validation(config)

model = SoundStream(config).to(device)

discriminator = FullDiscriminator().to(device)

g_optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"], betas=(0.5, 0.9))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["train"]["lr"], betas=(0.5, 0.9))

rec_loss = ReconstructionLoss(config).to(device)

start_step = 0
resume_from = config["checkpoint"].get("resume_from")
if resume_from:
    checkpoint = load_checkpoint(resume_from, model, discriminator, g_optimizer, d_optimizer, device)
    start_step = checkpoint["step"] + 1
else:
    initialize_codebook(model, train_loader, device, config["model"]["codebook_size"])

exp = None
if config["logging"]["comet"]:
    comet_ml.login()
    exp = comet_ml.Experiment(project_name=config["logging"]["project"])
    exp.set_name(config["logging"]["experiment"])
fixed_batch = next(iter(train_loader))[:config["logging"]["audio_count"]].to(device)
train(model, discriminator, train_loader, discriminator_loss, d_optimizer, generator_loss, g_optimizer,
      rec_loss, device, config, exp, start_step=start_step, fixed_batch=fixed_batch, val=val)

if exp is not None:
    exp.end()
