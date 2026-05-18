from tqdm.auto import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config, exp):
    model.train()

    avg_loss = 0
    step = epoch * len(dataloader)

    for batch_ind, waveform in tqdm(enumerate(dataloader), total=len(dataloader)):
        waveform = waveform.to(device)
        out = model(waveform)

        loss = criterion(waveform, out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()

        if exp is not None:
            exp.log_metrics(
                {"train_step_loss": loss.item()},
                step=step + batch_ind,
            )

        if config["train"]["batch_limit"] and batch_ind + 1 == config["train"]["batch_count"]:
            break

    avg_loss = avg_loss / len(dataloader)
    return {"avg_loss": avg_loss}

