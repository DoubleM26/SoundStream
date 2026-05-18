from tqdm.auto import tqdm
import torch

from src.checkpoint import checkpoint_paths, save_checkpoint


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
            file_name=f"audio/recon_{i}.wav",
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


def train(
        model,
        discriminator,
        dataloader,
        discriminator_loss,
        d_optimizer,
        generator_loss,
        g_optimizer,
        rec_loss,
        device,
        config,
        exp,
        start_step=0,
        fixed_batch=None,
        val=None
):
    model.train()
    discriminator.train()

    global_step = start_step
    max_steps = config["train"]["max_steps"]
    if max_steps is None:
        max_steps = config["train"].get("batch_count", len(dataloader)) if config["train"].get("batch_limit") else len(
            dataloader)

    progress = tqdm(total=max_steps - start_step)
    while global_step < max_steps:
        for waveform in dataloader:
            if global_step >= max_steps:
                break

            real_wf = waveform.to(device)

            with torch.no_grad():
                out = model(real_wf)
                fake_wf = out["recon"]

            real_preds = discriminator(real_wf)
            fake_preds = discriminator(fake_wf.detach())

            d_loss, d_loss_logs = discriminator_loss(real_preds, fake_preds, config)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            discriminator.requires_grad_(False)

            out = model(real_wf)
            fake_wf = out["recon"]
            real_preds = discriminator(real_wf.detach())
            fake_preds = discriminator(fake_wf)

            g_loss, g_loss_logs = generator_loss(real_wf, fake_wf, real_preds, fake_preds, rec_loss,
                                                 out["commitment_loss"], config)

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

            if val is not None and global_step % config["validation"]["every"] == 0:
                res = val.validate(model)
                val_metrics = {}
                for k, v in res.items():
                    val_metrics["val/" + k] = v

                if exp is not None:
                    exp.log_metrics(val_metrics, step=global_step)

            if exp is not None:
                if global_step % config["train"]["log_every"] == 0:
                    exp.log_metrics(metrics, step=global_step)

                if global_step % config["logging"]["audio_every"] == 0:
                    log_fixed_audio(
                        exp,
                        model,
                        fixed_batch,
                        config["data"]["sample_rate"],
                        global_step,
                        log_original=(global_step == 0),
                    )

            save_every = config["checkpoint"]["save_every"]
            keep_every = config["checkpoint"]["keep_every"]
            if save_every and (global_step + 1) % save_every == 0:
                latest_path, step_path = checkpoint_paths(config, global_step)
                save_checkpoint(latest_path, model, discriminator, g_optimizer, d_optimizer, global_step, config)
                if keep_every and (global_step + 1) % keep_every == 0:
                    save_checkpoint(step_path, model, discriminator, g_optimizer, d_optimizer, global_step, config)

            global_step += 1
            progress.update(1)

    res = val.validate(model)
    if exp is not None:
        exp.log_metrics(res, step=global_step)
    progress.close()

    latest_path, _ = checkpoint_paths(config, global_step - 1)
    save_checkpoint(latest_path, model, discriminator, g_optimizer, d_optimizer, global_step - 1, config)
