import torch
from torch.nn import functional as F

from models.fno import FNO_WOR
from models.mgno import MgNO
from physics.cbs import ConvergentBornSeries_Batch


def make_operator(kind, device):
    if kind == 'mgno':
        return MgNO(lifting=None, proj=None, dim_input=4, features=24, loss='rel_l2', learning_rate=0.0005, step_size=100, gamma=0.5, weight_decay=1e-5, eta_min=0.0005, iteration=6).to(device)
    if kind == 'fno':
        return FNO_WOR({'width': 32, 'modes': 25, 'n_layers': 4, 'retrain_fno': 1234}, device=device, padding_frac=0.25).to(device)
    raise ValueError(kind)


def normalize_speed(speed, minimum=1408.692, maximum=1595.1279):
    interior = F.interpolate(speed[:, :, 90:390, 90:390], size=(256, 256), mode='bilinear', align_corners=False)
    return 2 * (interior - minimum) / (maximum - minimum) - 1


def denormalize_speed(normalized, minimum=1408.692, maximum=1595.1279):
    interior = F.interpolate(normalized, size=(300, 300), mode='bilinear', align_corners=False)
    interior = (interior + 1) * (maximum - minimum) / 2 + minimum
    return F.pad(interior, (90, 90, 90, 90), value=1500)


@torch.no_grad()
def tweedie_speed(speed, score_fn, sde, time_min=0.1, time_max=0.2):
    normalized = normalize_speed(speed)
    time = torch.empty(speed.shape[0], device=speed.device).uniform_(time_min, time_max)
    time = (torch.floor(time * sde.N) + 1) / sde.N
    mean, std, coef = sde.marginal_prob(normalized, time)
    noisy = mean + std[:, None, None, None] * torch.randn_like(normalized)
    score = score_fn(noisy, time)
    estimate = (noisy + std[:, None, None, None].square() * score) / coef[:, None, None, None]
    return denormalize_speed(estimate)


@torch.no_grad()
def online_cbs_targets(speed, source_locations, source_indices, iterations=500):
    fields = []
    for batch_index, source_index in enumerate(source_indices.tolist()):
        source = source_locations[source_index:source_index + 1]
        solver = ConvergentBornSeries_Batch(f=500000, sos=speed[batch_index:batch_index + 1], boundary_width=[300, 300], boundary_strength=225, boundary_type='PML3', src_loc_set=source, device=speed.device)
        field = solver(max_iters=iterations)[:, 0]
        fields.append(torch.stack((field.real, field.imag), dim=1))
    return torch.cat(fields, dim=0)


def relative_l2(prediction, target):
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return (numerator / denominator).mean()


def train_operator_online(kind, data, source_locations, homogeneous_fields, epochs, device, score_fn=None, sde=None, model=None, learning_rate=0.0005):
    model = make_operator(kind, device) if model is None else model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    source_locations = torch.as_tensor(source_locations, device=device)
    homogeneous_fields = torch.as_tensor(homogeneous_fields, device=device)
    if torch.is_complex(homogeneous_fields):
        homogeneous_fields = torch.stack((homogeneous_fields.real, homogeneous_fields.imag), dim=1)
    if homogeneous_fields.shape[0] != source_locations.shape[0] or homogeneous_fields.shape[1] != 2:
        raise ValueError('homogeneous fields must have two channels per source')
    for _ in range(epochs):
        model.train()
        for speed in data:
            if isinstance(speed, (tuple, list)):
                speed = speed[0]
            speed = speed.to(device)
            if score_fn is not None:
                speed = tweedie_speed(speed, score_fn, sde)
            source_indices = torch.randint(len(source_locations), (speed.shape[0],), device=device)
            incident = homogeneous_fields.index_select(0, source_indices)
            target = online_cbs_targets(speed, source_locations, source_indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(speed, incident)
            loss = relative_l2(prediction, target)
            loss.backward()
            optimizer.step()
    return model
