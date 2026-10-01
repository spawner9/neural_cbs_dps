import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        return torch.einsum('bixy,ioxy->boxy', input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        comp_weights1 = torch.view_as_complex(self.weights1)
        comp_weights2 = torch.view_as_complex(self.weights2)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], comp_weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], comp_weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO_WOR(nn.Module):

    def __init__(self, fno_architecture, device=None, padding_frac=1 / 4):
        super(FNO_WOR, self).__init__()
        self.modes1 = fno_architecture['modes']
        self.modes2 = fno_architecture['modes']
        if self.modes2 > 35:
            self.modes2 = int(self.modes2 / 2)
        self.width = fno_architecture['width']
        self.n_layers = fno_architecture['n_layers']
        self.retrain_fno = fno_architecture['retrain_fno']
        self.fc0 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=1)
        (mean_sos, std_sos) = (torch.tensor(1488.3911, dtype=torch.float32), torch.tensor(27.5279, dtype=torch.float32))
        self.register_buffer('mean_sos', mean_sos)
        self.register_buffer('std_sos', std_sos)
        torch.manual_seed(self.retrain_fno)
        self.padding_frac = padding_frac
        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])
        self.fc1 = nn.Conv2d(in_channels=self.width, out_channels=128, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def forward(self, x, xs):
        x = (x - self.mean_sos) / (self.std_sos * 0.6)
        x = torch.cat([x, xs], dim=1)
        x = self.fc0(x)
        x1_padding = int(round(x.shape[-1] * self.padding_frac))
        x2_padding = int(round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])
        for (k, (s, c)) in enumerate(zip(self.spectral_list, self.conv_list)):
            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        x = x[..., :-x1_padding, :-x2_padding]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
