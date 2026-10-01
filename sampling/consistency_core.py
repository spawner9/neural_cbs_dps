import math

import torch

from sampling.common import physics_update
from training.edm_consistency_control import karras_sigma


@torch.no_grad()
def sample_consistency(observed, inversion_input, backbone, control, inversion_net, diffusion, bank, schedule=(0, 10, 20, 25, 30, 35, 39), step_size=0.4, source_batch=16, generator=None):
    if len(schedule) < 2 or list(schedule) != sorted(set(schedule)):
        raise ValueError('schedule must contain ordered unique indices')
    hint = inversion_net(inversion_input)
    noise = torch.randn((observed.shape[0], 1, 256, 256), device=observed.device, generator=generator)
    state = noise * diffusion.sigma_max
    for position, index in enumerate(schedule[:-1]):
        sigma = torch.full((state.shape[0],), karras_sigma(index, levels=40), device=state.device)
        estimate = diffusion.recon(backbone, control, state, hint, sigma)[1].clamp(-1, 1)
        if position == len(schedule) - 2:
            return estimate
        corrected = physics_update(estimate, observed, bank, step_size, source_batch)
        next_sigma = karras_sigma(schedule[position + 1], levels=40)
        scale = math.sqrt(max(next_sigma**2 - diffusion.sigma_min**2, 0))
        state = corrected + torch.randn(corrected.shape, device=corrected.device, generator=generator) * scale
