import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvergentBornSeries(nn.Module):

    def __init__(self, lamb=1, sos=None, src_loc=[30, 60], boundary_width=[8, 8], boundary_strength=1, boundary_type='PML3', device='cuda:0'):
        super().__init__()
        self.device = device
        self.lamb = lamb
        self.omega0 = 2 * torch.pi / self.lamb
        self.PPW = 4
        self.pixel_size = self.lamb / self.PPW
        self.sos = nn.Parameter(sos, requires_grad=True).to(device)
        self.N = torch.tensor(sos.size()).to(int)
        self.src = torch.zeros_like(self.sos)
        self.src[src_loc[0], src_loc[1]] = 1.0
        self.src = nn.Parameter(self.src, requires_grad=False)
        self.boundary_widths = torch.tensor(boundary_width, dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()
        self.k = self.omega0 / self.sos
        self.k0 = torch.sqrt((torch.max(self.k ** 2) + torch.min(self.k ** 2)) / 2).detach()
        self.k2_pad = self.set_boundary(self.k)
        self.epsilon = torch.max(torch.abs(self.k2_pad - self.k0 ** 2)).detach()
        self.g0 = 1 / (self.px_range ** 2 + self.py_range ** 2 - self.k0 ** 2 - 1j * self.epsilon).detach()
        self.V = self.k2_pad - self.k0 ** 2 - 1j * self.epsilon
        self.gamma = 1j / self.epsilon * self.V

    def forward(self, max_iters=5, tol=1e-16, requries_grad=False):
        self.u = torch.zeros(self.new_N[0], self.new_N[1]).to(self.V.device)
        if requries_grad:
            for _ in range(max_iters):
                tmp1 = self.V * self.u
                tmp1[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src
                tmp2 = self.gamma * (self.u - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                self.u = self.u - tmp2
        else:
            with torch.no_grad():
                for _ in range(max_iters):
                    tmp1 = self.V * self.u
                    tmp1[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src
                    tmp2 = self.gamma * (self.u - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                    self.u = self.u - tmp2
        return self.u[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]

    def set_grid(self):
        self.new_N = (2 ** torch.ceil(torch.log2(self.N + self.boundary_widths))).to(int).to(self.device)
        self.x_range = (torch.arange(self.new_N[0]) * self.pixel_size).view(-1, 1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1]) * self.pixel_size).to(self.device)
        self.px_range = 2 * torch.pi * torch.fft.fftfreq(self.new_N[0], d=self.pixel_size).view(-1, 1).to(self.device)
        self.py_range = (2 * torch.pi * torch.fft.fftfreq(self.new_N[1], d=self.pixel_size)).to(self.device)
        self.roi_size = torch.tensor(self.sos.size()).to(self.device)
        self.Bl = torch.ceil((self.new_N - self.roi_size) / 2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size) / 2).to(int)
        self.roi = [[self.Bl[0], self.Bl[0] + self.roi_size[0]], [self.Bl[1], self.Bl[1] + self.roi_size[1]]]
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1, 1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        self.dist = (torch.sqrt(self.x ** 2 + self.y ** 2) * self.pixel_size).to(self.device)

    def set_boundary(self, k):
        k_pad = F.pad(k.unsqueeze(0), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='replicate').squeeze()
        k0 = torch.sqrt(torch.mean(k_pad ** 2))
        c = self.boundary_strength * k0 ** 2 / (2 * k0)
        (f_boundary_curve, _) = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type, r=self.dist, Bmax=self.Bmax, c=c, k0=k0)
        k2_pad = k_pad ** 2 + (f_boundary_curve * self.omega0 ** 2).detach()
        return k2_pad

    def compute_leakage_and_boundary_curve(self, boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5':
            f_boundary_curve = 1 / k0 ** 2 * (c ** 7 * r ** 5 * (6.0 + (2j * k0 - c) * r)) / (720 + 720 * c * r + 360 * c ** 2 * r ** 2 + 120 * c ** 3 * r ** 3 + 30 * c ** 4 * r ** 4 + 6 * c ** 5 * r ** 5 + c ** 6 * r ** 6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c ** 2 * Bmax ** 2 + 120 * c ** 3 * Bmax ** 3 + 30 * c ** 4 * Bmax ** 4 + 6 * c ** 5 * Bmax ** 5 + c ** 6 * Bmax ** 6) / 24
        elif boundary_type == 'PML4':
            f_boundary_curve = 1 / k0 ** 2 * (c ** 6 * r ** 4 * (5.0 + (2j * k0 - c) * r)) / (120 + 120 * c * r + 60 * c ** 2 * r ** 2 + 20 * c ** 3 * r ** 3 + 5 * c ** 4 * r ** 4 + c ** 5 * r ** 5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c ** 2 * Bmax ** 2 + 20 * c ** 3 * Bmax ** 3 + 5 * c ** 4 * Bmax ** 4 + c ** 5 * Bmax ** 5) / 24
        elif boundary_type == 'PML3':
            f_boundary_curve = 1 / k0 ** 2 * (c ** 5 * r ** 3 * (4.0 + (2j * k0 - c) * r)) / (24 + 24 * c * r + 12 * c ** 2 * r ** 2 + 4 * c ** 3 * r ** 3 + c ** 4 * r ** 4)
            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c ** 2 * Bmax ** 2 + 4 * c ** 3 * Bmax ** 3 + c ** 4 * Bmax ** 4) / 24
        elif boundary_type == 'PML2':
            f_boundary_curve = 1 / k0 ** 2 * (c ** 4 * r ** 2 * (3.0 + (2j * k0 - c) * r)) / (6 + 6 * c * r + 3 * c ** 2 * r ** 2 + c ** 3 * r ** 3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c ** 2 * Bmax ** 2 + c ** 3 * Bmax ** 3) / 6
        elif boundary_type == 'PML1':
            f_boundary_curve = 1 / k0 ** 2 * (c ** 3 * r * (2.0 + (2j * k0 - c) * r)) / (2.0 + 2.0 * c * r + c ** 2 * r ** 2) / k0 ** 2
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c ** 2 * Bmax ** 2) / 2
        else:
            raise ValueError(f'Unknown boundary type: {boundary_type}')
        return (f_boundary_curve, leakage)

    def get_u(self):
        return self.u[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]].detach().cpu()

    def set_src_loc(self, src_loc=[30, 60]):
        self.src.data = torch.zeros_like(self.sos).to(self.V.device)
        self.src.data[src_loc[0], src_loc[1]] = 1.0

class ConvergentBornSeries_Batch(nn.Module):

    def __init__(self, f=1, sos=None, boundary_width=[8, 8], boundary_strength=1, boundary_type='PML3', src_loc_set=np.array([[30, 60], [240, 280]]), device='cuda:0'):
        super().__init__()
        self.device = device
        self.lamb = 1500 / f
        self.omega0 = 2 * torch.pi * f
        self.PPW = 6
        self.pixel_size = self.lamb / self.PPW
        self.sos = sos
        self.N = torch.tensor(sos.size()[2:]).to(int)
        self.src_set = self.set_src_loc(src_loc_set)
        self.boundary_widths = torch.tensor(boundary_width, dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()
        self.k = self.omega0 / self.sos
        self.k0 = torch.sqrt((torch.amax(self.k ** 2, dim=(2, 3)) + torch.amin(self.k ** 2, dim=(2, 3))) / 2)[:, :, None, None]
        self.k2_pad = self.set_boundary(self.k)
        self.epsilon = torch.amax(torch.abs(self.k2_pad - self.k0 ** 2), dim=(2, 3))[:, :, None, None]
        self.g0 = 1 / (self.pxy_grid ** 2 - self.k0 ** 2 - 1j * self.epsilon)
        self.V = self.k2_pad - self.k0 ** 2 - 1j * self.epsilon
        self.gamma = 1j / self.epsilon * self.V

    def forward(self, max_iters=5, tol=1e-16):
        self.sample_size = self.src_set.shape[1]
        self.u = torch.zeros_like(self.k2_pad).to(self.device)
        self.u_batch = self.u.repeat(1, self.sample_size, 1, 1)
        with torch.no_grad():
            for _ in range(max_iters):
                tmp1 = self.V * self.u_batch
                tmp1[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src_set
                tmp2 = self.gamma * (self.u_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1, dim=(-2, -1)), dim=(-2, -1)))
                self.u_batch = self.u_batch - tmp2
        return self.u_batch[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]

    def set_grid(self):
        self.new_N = (2 ** torch.ceil(torch.log2(self.N + self.boundary_widths))).to(int).to(self.device)
        self.x_range = (torch.arange(self.new_N[0]) * self.pixel_size).view(-1, 1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1]) * self.pixel_size).to(self.device)
        self.px_range = 2 * torch.pi * torch.fft.fftfreq(self.new_N[0], d=self.pixel_size).view(-1, 1).to(self.device)
        self.py_range = (2 * torch.pi * torch.fft.fftfreq(self.new_N[1], d=self.pixel_size)).to(self.device)
        self.pxy_grid = torch.sqrt(self.px_range ** 2 + self.py_range ** 2).to(self.device)[None, None, :, :]
        self.roi_size = torch.tensor(self.sos.size()[2:]).to(self.device)
        self.Bl = torch.ceil((self.new_N - self.roi_size) / 2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size) / 2).to(int)
        self.roi = [[self.Bl[0], self.Bl[0] + self.roi_size[0]], [self.Bl[1], self.Bl[1] + self.roi_size[1]]]
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1, 1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        self.dist = (torch.sqrt(self.x ** 2 + self.y ** 2) * self.pixel_size).to(self.device)[None, None, :, :]

    def k_pad_batch(self, k):
        k_pad = F.pad(torch.zeros_like(k), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='constant', value=0.0)
        k_pad = k_pad + self.k0
        k_pad[..., self.Bl[0]:self.Bl[0] + k.size(2), self.Bl[1]:self.Bl[1] + k.size(3)] = k
        return k_pad

    def set_boundary(self, k):
        (f_boundary_curve, _) = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type, r=self.dist, Bmax=self.Bmax, c=self.boundary_strength, k0=self.k0)
        k_pad = self.k_pad_batch(k)
        k2_pad = k_pad ** 2 + f_boundary_curve
        return k2_pad

    def compute_leakage_and_boundary_curve(self, boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5':
            f_boundary_curve = c ** 7 * r ** 5 * (6.0 + (2j * k0 - c) * r) / (720 + 720 * c * r + 360 * c ** 2 * r ** 2 + 120 * c ** 3 * r ** 3 + 30 * c ** 4 * r ** 4 + 6 * c ** 5 * r ** 5 + c ** 6 * r ** 6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c ** 2 * Bmax ** 2 + 120 * c ** 3 * Bmax ** 3 + 30 * c ** 4 * Bmax ** 4 + 6 * c ** 5 * Bmax ** 5 + c ** 6 * Bmax ** 6) / 24
        elif boundary_type == 'PML4':
            f_boundary_curve = c ** 6 * r ** 4 * (5.0 + (2j * k0 - c) * r) / (120 + 120 * c * r + 60 * c ** 2 * r ** 2 + 20 * c ** 3 * r ** 3 + 5 * c ** 4 * r ** 4 + c ** 5 * r ** 5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c ** 2 * Bmax ** 2 + 20 * c ** 3 * Bmax ** 3 + 5 * c ** 4 * Bmax ** 4 + c ** 5 * Bmax ** 5) / 24
        elif boundary_type == 'PML3':
            f_boundary_curve = c ** 5 * r ** 3 * (4.0 + (2j * k0 - c) * r) / (24 + 24 * c * r + 12 * c ** 2 * r ** 2 + 4 * c ** 3 * r ** 3 + c ** 4 * r ** 4)
            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c ** 2 * Bmax ** 2 + 4 * c ** 3 * Bmax ** 3 + c ** 4 * Bmax ** 4) / 24
        elif boundary_type == 'PML2':
            f_boundary_curve = c ** 4 * r ** 2 * (3.0 + (2j * k0 - c) * r) / (6 + 6 * c * r + 3 * c ** 2 * r ** 2 + c ** 3 * r ** 3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c ** 2 * Bmax ** 2 + c ** 3 * Bmax ** 3) / 6
        elif boundary_type == 'PML1':
            f_boundary_curve = c ** 3 * r * (2.0 + (2j * k0 - c) * r) / (2.0 + 2.0 * c * r + c ** 2 * r ** 2) / k0 ** 2
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c ** 2 * Bmax ** 2) / 2
        else:
            raise ValueError(f'Unknown boundary type: {boundary_type}')
        return (f_boundary_curve, leakage)

    def set_src_loc(self, src_loc_set):
        sample_size = src_loc_set.shape[0]
        src = torch.zeros_like(self.sos[0], dtype=torch.complex64).to(self.device)
        src_set = torch.stack([src] * sample_size, dim=1)
        for i in range(sample_size):
            src_set[0, i, src_loc_set[i, 0], src_loc_set[i, 1]] = torch.tensor(10000000.0)
        return src_set

class ConvergentBornSeries_Batch_Adjoint(ConvergentBornSeries_Batch):

    def __init__(self, batch_model, rec_loc, dobs_500k_batch, dobs_500k_mask):
        self.__dict__ = batch_model.__dict__.copy()
        self.dobs_500k_batch = dobs_500k_batch.to(batch_model.device)
        self.dobs_500k_mask = torch.tensor(dobs_500k_mask).to(batch_model.device)
        self.rec_loc = rec_loc

    def forward(self, u_batch, max_iters=5):
        (self.batch_size, self.sample_size) = (u_batch.shape[0], u_batch.shape[1])
        (self.rec_diff_batch, rec_diff_value) = self.compute_rec_diff(u_batch)
        self.adjoint_lamb = torch.zeros_like(self.k2_pad[0:1, :, :, :]).to(self.device)
        self.adjoint_lamb_batch = self.adjoint_lamb.repeat(self.batch_size, self.sample_size, 1, 1)
        with torch.no_grad():
            for _ in range(max_iters):
                tmp1 = self.V * self.adjoint_lamb_batch
                tmp1[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += torch.conj(self.rec_diff_batch)
                tmp2 = self.gamma * (self.adjoint_lamb_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1, dim=(-2, -1)), dim=(-2, -1)))
                self.adjoint_lamb_batch = self.adjoint_lamb_batch - tmp2
        self.real_inner_prod_sum = torch.real(torch.sum(self.adjoint_lamb_batch[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] * u_batch, dim=1))[:, None, :, :]
        self.sos_grad = -2 * (self.omega0 ** 2 / self.sos ** 3) * self.real_inner_prod_sum
        return (self.sos_grad, rec_diff_value)

    def compute_rec_diff(self, u_batch):
        rec_diff_batch = torch.zeros_like(u_batch).to(self.device)
        self.rec_diff = -self.dobs_500k_batch + u_batch[:, :, self.rec_loc[:, 0], self.rec_loc[:, 1]] * self.dobs_500k_mask
        rec_diff_batch[:, :, self.rec_loc[:, 0], self.rec_loc[:, 1]] = self.rec_diff * 10000000.0
        return (rec_diff_batch, torch.mean(torch.abs(self.rec_diff)))
