import torch

from sampling.common import physics_update
from sampling.physics_bank import CBSBank


def tweedie(noisy, score, standard_deviation, coefficient):
    return (noisy + standard_deviation[:, None, None, None].square() * score) / coefficient[:, None, None, None]


@torch.no_grad()
def sample_dps_ddim_cbs(initial, observed, score_fn, sde, geometry, times, eta, step_size=0.1, source_batch=16):
    if len(times) < 2 or any(times[i] <= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError('times must be strictly decreasing')
    if not 0 <= eta <= 1:
        raise ValueError('eta must lie in [0, 1]')
    bank = CBSBank(geometry)
    estimate = physics_update(initial, observed, bank, step_size, source_batch)
    time = torch.full((initial.shape[0],), times[0], device=initial.device)
    mean, std, coef = sde.marginal_prob(estimate, time)
    noisy = mean + std[:, None, None, None] * torch.randn_like(estimate)
    score = score_fn(noisy, time)
    estimate = tweedie(noisy, score, std, coef)
    for value in times[1:-1]:
        corrected = physics_update(estimate, observed, bank, step_size, source_batch)
        next_time = torch.full_like(time, value)
        mean_next, std_next, coef_next = sde.marginal_prob(corrected, next_time)
        stochastic_scale = (std_next / std) * (1 - (coef / coef_next).square()).clamp_min(0).sqrt()
        deterministic_scale = (std_next.square() - (eta * stochastic_scale).square()).clamp_min(0).sqrt()
        estimated_noise = -score * std[:, None, None, None]
        noisy = mean_next + deterministic_scale[:, None, None, None] * estimated_noise + (eta * stochastic_scale)[:, None, None, None] * torch.randn_like(noisy)
        time = next_time
        std = std_next
        coef = coef_next
        score = score_fn(noisy, time)
        estimate = tweedie(noisy, score, std, coef)
    final_time = torch.full_like(time, times[-1])
    mean, std, coef = sde.marginal_prob(estimate, final_time)
    noisy = mean + std[:, None, None, None] * torch.randn_like(estimate)
    return tweedie(noisy, score_fn(noisy, final_time), std, coef)


def sample_dps_cbs(initial, observed, score_fn, sde, geometry, times, step_size=0.1, source_batch=16):
    return sample_dps_ddim_cbs(initial, observed, score_fn, sde, geometry, times, eta=1.0, step_size=step_size, source_batch=source_batch)


def sample_ddim_cbs(initial, observed, score_fn, sde, geometry, times, step_size=0.1, source_batch=16):
    return sample_dps_ddim_cbs(initial, observed, score_fn, sde, geometry, times, eta=0.0, step_size=step_size, source_batch=source_batch)
