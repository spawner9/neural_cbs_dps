import torch


def karras_sigma(index, levels=40, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    return (sigma_max ** (1 / rho) + index / (levels - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def controlled_reconstruction(backbone, control, noisy, hint, sigma, sigma_data=0.5, sigma_min=0.002):
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_skip = c_skip[:, None, None, None]
    c_out = c_out[:, None, None, None]
    c_in = c_in[:, None, None, None]
    time = 250 * torch.log(sigma.clamp_min(1e-44))
    features = control(c_in * noisy, c_in * hint, time)
    estimate = backbone(x=c_in * noisy, timesteps=time, control=features)
    return c_skip * noisy + c_out * estimate


def control_loss(backbone, control, target_control, clean, hint, levels=40, objective='reconstruction'):
    batch = clean.shape[0]
    index = torch.randint(0, levels - 1, (batch,), device=clean.device)
    sigma = karras_sigma(index, levels)
    noise = torch.randn_like(clean)
    noisy = clean + sigma[:, None, None, None] * noise
    estimate = controlled_reconstruction(backbone, control, noisy, hint, sigma)
    if objective == 'reconstruction':
        target = clean
    elif objective == 'consistency':
        if target_control is None:
            raise ValueError('consistency training requires a target control model')
        next_sigma = karras_sigma(index + 1, levels)
        next_noisy = clean + next_sigma[:, None, None, None] * noise
        with torch.no_grad():
            target = controlled_reconstruction(backbone, target_control, next_noisy, hint, next_sigma)
    else:
        raise ValueError(objective)
    return (estimate - target).square().flatten(1).mean(1).mean()


def train_control(backbone, control, hint_net, data, optimizer, steps, device, target_control=None, objective='reconstruction', ema_decay=0.9999):
    backbone.eval()
    backbone.requires_grad_(False)
    hint_net.eval()
    hint_net.requires_grad_(False)
    control.train()
    iterator = iter(data)
    for _ in range(steps):
        try:
            observation, clean = next(iterator)
        except StopIteration:
            iterator = iter(data)
            observation, clean = next(iterator)
        observation = observation.to(device)
        clean = clean.to(device)
        with torch.no_grad():
            hint = hint_net(observation)
        optimizer.zero_grad(set_to_none=True)
        loss = control_loss(backbone, control, target_control, clean, hint, objective=objective)
        loss.backward()
        optimizer.step()
        if target_control is not None:
            with torch.no_grad():
                for target, source in zip(target_control.parameters(), control.parameters()):
                    target.lerp_(source, 1 - ema_decay)
    return control
