from tqdm.auto import tqdm
import torch


def log_fixed_audio(exp, model, fixed_wf, sample_rate, step, log_original=False):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(fixed_wf)
        recon = out["recon"]
    real = fixed_wf.detach().cpu().clamp(-1, 1)
    fake = recon.detach().cpu().clamp(-1, 1)
    for i in range(real.shape[0]):
        if log_original:
            exp.log_audio(
                real[i, 0].numpy(),
                sample_rate=sample_rate,
                file_name=f"audio/original_{i}.wav",
                step=step,
            )
        exp.log_audio(
            fake[i, 0].numpy(),
            sample_rate=sample_rate,
            file_name=f"audio/recon_{i}/step_{step}.wav",
            step=step,
        )
    model.train(was_training)


@torch.no_grad()
def initialize_codebook(model, dataloader, device, codebook_size):
    model.train()

    data = []
    vectors = 0
    for waveform in dataloader:
        waveform = waveform.to(device)
        encoded = model.encoder(waveform)
        data.append(encoded)
        vectors += encoded.shape[0] * encoded.shape[-1]
        if vectors >= codebook_size:
            break

    model.quantizer.initialize(torch.cat(data, dim=0))


def train_one_epoch(
        model,
        discriminator,
        dataloader,
        discriminator_loss,
        d_optimizer,
        generator_loss,
        g_optimizer,
        rec_loss,
        device,
        epoch,
        config,
        exp
):
    model.train()

    g_avg_loss = 0
    d_avg_loss = 0
    step = epoch * len(dataloader)

    for batch_ind, waveform in tqdm(enumerate(dataloader), total=len(dataloader)):
        real_wf = waveform.to(device)

        with torch.no_grad():
            out = model(real_wf)
            fake_wf = out["recon"]

        real_preds = discriminator(real_wf)
        fake_preds = discriminator(fake_wf.detach())

        d_loss, d_loss_logs = discriminator_loss(real_preds, fake_preds, config)
        d_avg_loss += d_loss.item()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        discriminator.requires_grad_(False)

        out = model(real_wf)
        fake_wf = out["recon"]
        real_preds = discriminator(real_wf.detach())
        fake_preds = discriminator(fake_wf)

        g_loss, g_loss_logs = generator_loss(real_wf, fake_wf, real_preds, fake_preds, rec_loss, out["commitment_loss"],
                                             config)
        g_avg_loss += g_loss.item()

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        discriminator.requires_grad_(True)

        metrics = {}
        for k, v in d_loss_logs.items():
            metrics["d/" + k] = v
        for k, v in g_loss_logs.items():
            metrics["g/" + k] = v

        for k, v in out["rvq_stats"].items():
            metrics["rvq/" + k] = v

        if exp is not None:
            exp.log_metrics(
                metrics,
                step=step + batch_ind,
            )

            if (step + batch_ind) % config["logging"]["audio_every"] == 0:
                log_fixed_audio(
                    exp,
                    model,
                    config["logging"]["fixed_batch"],
                    config["data"]["sample_rate"],
                    step + batch_ind,
                    log_original=(step + batch_ind == 0),
                )

        if config["train"]["batch_limit"] and batch_ind + 1 == config["train"]["batch_count"]:
            break

    g_avg_loss /= batch_ind + 1
    d_avg_loss /= batch_ind + 1
    return {
        "g_avg_loss": g_avg_loss,
        "d_avg_loss": d_avg_loss
    }
