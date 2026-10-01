import torch
from torch.nn import functional as F


def denormalize_speed(normalized, minimum=1408.692, maximum=1595.1279):
    interior = F.interpolate(normalized, size=(300, 300), mode='bilinear', align_corners=False)
    interior = (interior + 1) * (maximum - minimum) / 2 + minimum
    return F.pad(interior, (90, 90, 90, 90), value=1500)


def normalize_speed(speed, minimum=1408.692, maximum=1595.1279):
    interior = F.interpolate(speed[:, :, 90:390, 90:390], size=(256, 256), mode='bilinear', align_corners=False)
    return 2 * (interior - minimum) / (maximum - minimum) - 1


@torch.no_grad()
def physics_update(normalized, observed, bank, step_size, source_batch=16):
    speed = denormalize_speed(normalized)
    gradient, residual = bank.gradient(speed, observed, source_batch)
    update = step_size * gradient / residual.clamp_min(1e-30).sqrt()[:, None, None, None]
    speed[:, :, 90:390, 90:390] -= update[:, :, 90:390, 90:390]
    speed[:, :, 90:390, 90:390].clamp_(1408.692, 1595.1279)
    return normalize_speed(speed)
