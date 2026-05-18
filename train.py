import torch
from torch.utils.data import DataLoader
import comet_ml

from src.data import SoundStreamDataset
from src.discriminator import FullDiscriminator
from src.loops import train_one_epoch, initialize_codebook
from src.config import load_config
from src.losses import discriminator_loss, generator_loss, ReconstructionLoss
from src.model.model import SoundStream

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
model = SoundStream(config).to(device)

discriminator = FullDiscriminator().to(device)

g_optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"], betas=(0.5, 0.9))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["train"]["lr"], betas=(0.5, 0.9))

rec_loss = ReconstructionLoss(config).to(device)

initialize_codebook(model, train_loader, device, config["model"]["codebook_size"])

exp = None
if config["logging"]["comet"]:
    comet_ml.login()
    exp = comet_ml.Experiment(project_name=config["logging"]["project"])
    exp.set_name(config["logging"]["experiment"])
fixed_batch = next(iter(train_loader))[:config["logging"]["audio_count"]].to(device)
config["logging"]["fixed_batch"] = fixed_batch
train_one_epoch(model, discriminator, train_loader, discriminator_loss, d_optimizer, generator_loss, g_optimizer,
                rec_loss,
                device, 0, config, exp)

if exp is not None:
    exp.end()
