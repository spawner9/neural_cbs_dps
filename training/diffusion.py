import torch


def edm_loss(model, clean, sigma_data=0.5, p_mean=-1.2, p_std=1.2):
    sigma = (torch.randn(clean.shape[0], device=clean.device) * p_std + p_mean).exp()
    scale = sigma[:, None, None, None]
    noisy = clean + torch.randn_like(clean) * scale
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    time = 250 * torch.log(sigma.clamp_min(1e-44))
    prediction = model(c_in[:, None, None, None] * noisy, time)
    denoised = c_skip[:, None, None, None] * noisy + c_out[:, None, None, None] * prediction
    weight = sigma.reciprocal().square() + sigma_data**-2
    return (weight[:, None, None, None] * (denoised - clean).square()).flatten(1).mean(1).mean()


def train_diffusion(model, data, optimizer, steps, device, ema=None, ema_decay=0.9999):
    iterator = iter(data)
    model.train()
    for _ in range(steps):
        try:
            clean = next(iterator)
        except StopIteration:
            iterator = iter(data)
            clean = next(iterator)
        if isinstance(clean, (tuple, list)):
            clean = clean[0]
        clean = clean.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = edm_loss(model, clean)
        loss.backward()
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - ema_decay)
    return model
