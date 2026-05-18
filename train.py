import torch
from torch.utils.data import DataLoader
import comet_ml

from src.data import SoundStreamDataset
from src.model import Simple
from src.discriminator import FullDiscriminator
from src.loops import train_one_epoch
from src.config import load_config
from src.losses import discriminator_loss, generator_loss, ReconstructionLoss

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
model = Simple().to(device)

discriminator = FullDiscriminator().to(device)

g_optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])
d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config["train"]["lr"])

rec_loss = ReconstructionLoss(config).to(device)

exp = None
if config["logging"]["comet"]:
    comet_ml.login()
    exp = comet_ml.Experiment(project_name=config["logging"]["project"])
    exp.set_name(config["logging"]["experiment"])

train_one_epoch(model, discriminator, train_loader, discriminator_loss, d_optimizer, generator_loss, g_optimizer, rec_loss,
                device, 0, config, exp)
