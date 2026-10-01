from sampling.consistency_core import sample_consistency
from sampling.physics_bank import ANOBank


def sample_diff_ano(observed, inversion_input, backbone, control, inversion_net, diffusion, geometry, mgno, homogeneous_fields, **kwargs):
    bank = ANOBank(geometry, mgno, homogeneous_fields)
    return sample_consistency(observed, inversion_input, backbone, control, inversion_net, diffusion, bank, **kwargs)
