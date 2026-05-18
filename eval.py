import argparse
import torch

from src.config import load_config
from src.model.model import SoundStream
from src.validation import Validation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default="./configs/debug.yml")
    args = parser.parse_args()

    config = load_config(args.config_file)
    device = config["eval"]["device"]

    val = Validation(config, config["eval"]["name"])
    checkpoint = torch.load(config["eval"]["checkpoint_path"], map_location=device)
    model = SoundStream(checkpoint.get("config", config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    res = val.validate(model)

    print(res)


if __name__ == "__main__":
    main()
