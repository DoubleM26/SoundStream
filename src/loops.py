from tqdm.auto import tqdm


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

        fake_wf = model(real_wf.detach())

        real_preds = discriminator(real_wf)
        fake_preds = discriminator(fake_wf.detach())

        d_loss, d_loss_logs = discriminator_loss(real_preds, fake_preds, config)
        d_avg_loss += d_loss.item()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        discriminator.requires_grad_(False)
        fake_wf = model(real_wf)
        real_preds = discriminator(real_wf.detach())
        fake_preds = discriminator(fake_wf)

        g_loss, g_loss_logs = generator_loss(real_wf, fake_wf, real_preds, fake_preds, rec_loss, config)
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

        if exp is not None:
            exp.log_metrics(
                metrics,
                step=step + batch_ind,
            )

        if config["train"]["batch_limit"] and batch_ind + 1 == config["train"]["batch_count"]:
            break

    g_avg_loss /= len(dataloader)
    d_avg_loss /= len(dataloader)
    return {
        "g_avg_loss": g_avg_loss,
        "d_avg_loss": d_avg_loss
    }
