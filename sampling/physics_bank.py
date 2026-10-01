import torch

from physics.cbs import ConvergentBornSeries_Batch


class Geometry:
    def __init__(self, transmitters, receivers, mask):
        self.transmitters = torch.as_tensor(transmitters, dtype=torch.long)
        self.receivers = torch.as_tensor(receivers, dtype=torch.long)
        self.mask = torch.as_tensor(mask)

    def union(self):
        locations = []
        positions = {}
        transmitter_rows = []
        receiver_rows = []
        for collection, rows in ((self.transmitters, transmitter_rows), (self.receivers, receiver_rows)):
            for location in collection.tolist():
                key = tuple(location)
                if key not in positions:
                    positions[key] = len(locations)
                    locations.append(location)
                rows.append(positions[key])
        return torch.as_tensor(locations, dtype=torch.long), transmitter_rows, receiver_rows


class FieldBank:
    def __init__(self, geometry, frequency=500000.0, source_amplitude=1e7):
        self.geometry = geometry
        self.frequency = frequency
        self.source_amplitude = source_amplitude
        self.locations, self.transmitter_rows, self.receiver_rows = geometry.union()

    def gradient(self, speed, observed, source_batch=16):
        observed = observed.to(speed.device)
        fields = self.fields(speed, source_batch)
        transmitter_fields = fields[:, self.transmitter_rows]
        receiver_fields = fields[:, self.receiver_rows]
        receivers = self.geometry.receivers.to(speed.device)
        predicted = transmitter_fields[:, :, receivers[:, 0], receivers[:, 1]]
        mask = self.geometry.mask.to(speed.device)[None]
        residual = (predicted - observed) * mask
        adjoint = torch.einsum('bml,blhw->bmhw', residual.conj(), receiver_fields)
        omega = 2 * torch.pi * self.frequency
        gradient = -2 * omega**2 / speed**3 * (adjoint * transmitter_fields).sum(dim=1, keepdim=True).real
        mean_absolute_residual = residual.abs().mean(dim=(1, 2))
        return gradient, mean_absolute_residual


class CBSBank(FieldBank):
    @torch.no_grad()
    def fields(self, speed, source_batch=16):
        fields = []
        locations = self.locations.to(speed.device)
        for start in range(0, len(locations), source_batch):
            subset = locations[start:start + source_batch]
            solver = ConvergentBornSeries_Batch(f=self.frequency, sos=speed, boundary_width=[300, 300], boundary_strength=225, boundary_type='PML3', src_loc_set=subset, device=speed.device)
            fields.append(solver(max_iters=500))
        return torch.cat(fields, dim=1)


class ANOBank(FieldBank):
    def __init__(self, geometry, model, homogeneous_fields, frequency=500000.0):
        super().__init__(geometry, frequency)
        self.model = model
        self.homogeneous_fields = torch.as_tensor(homogeneous_fields)
        if torch.is_complex(self.homogeneous_fields):
            self.homogeneous_fields = torch.stack((self.homogeneous_fields.real, self.homogeneous_fields.imag), dim=1)
        if len(self.homogeneous_fields) != len(self.locations):
            raise ValueError('homogeneous fields must follow the source union')
        if self.homogeneous_fields.shape[1] != 2:
            raise ValueError('homogeneous fields need real and imaginary channels')

    @torch.no_grad()
    def fields(self, speed, source_batch=16):
        batch = speed.shape[0]
        homogeneous = self.homogeneous_fields.to(speed.device)
        outputs = []
        for start in range(0, len(homogeneous), source_batch):
            stop = min(start + source_batch, len(homogeneous))
            count = stop - start
            sound_speed = speed[:, None].expand(batch, count, 1, *speed.shape[-2:]).reshape(batch * count, 1, *speed.shape[-2:])
            incident = homogeneous[None, start:stop].expand(batch, count, 2, *speed.shape[-2:]).reshape(batch * count, 2, *speed.shape[-2:])
            prediction = self.model(sound_speed, incident).reshape(batch, count, 2, *speed.shape[-2:])
            outputs.append(torch.complex(prediction[:, :, 0], prediction[:, :, 1]))
        return torch.cat(outputs, dim=1)
