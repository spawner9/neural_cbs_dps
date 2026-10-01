from sampling.consistency_core import sample_consistency
from sampling.physics_bank import CBSBank


def sample_diff_cbs(observed, inversion_input, backbone, control, inversion_net, diffusion, geometry, **kwargs):
    bank = CBSBank(geometry)
    return sample_consistency(observed, inversion_input, backbone, control, inversion_net, diffusion, bank, **kwargs)
