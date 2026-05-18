import torch
from torch.utils.data import DataLoader
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment as NISQA
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from tqdm.auto import tqdm

from src.data import EvalSoundStreamDataset


def to_scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


class Validation:
    def __init__(self, config, data_name):
        self.device = config["train"]["device"]
        self.stoi = STOI(config["data"]["sample_rate"], False).to(self.device)
        self.nisqa = NISQA(config["data"]["sample_rate"]).to(self.device)
        val_dataset = EvalSoundStreamDataset("./", config["data"]["sample_rate"], data_name)
        self.dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    @torch.no_grad()
    def validate(self, model):
        was_training = model.training
        model.eval()

        stoi_data = []
        nisqa_data = []
        for _, batch in tqdm(self.dataloader, total=len(self.dataloader)):
            audio = batch.to(self.device)
            out = model(audio)
            recon = out["recon"].clamp(-1, 1)
            stoi_data.append(to_scalar(self.stoi(recon.squeeze(1), audio.squeeze(1))))
            nisqa_data.append(to_scalar(self.nisqa(recon.squeeze(1))[0]))
        model.train(was_training)
        return {
            "stoi": torch.tensor(stoi_data).mean(),
            "nisqa": torch.tensor(nisqa_data).mean()
        }
