import torch
from torch import nn
from torch.nn import functional as F

class LpLoss(nn.Module):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = h ** (self.d / self.p) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class RRMSE(object):

    def __init__(self):
        super(RRMSE, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        norm = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), 2, 1) ** 2
        normy = torch.norm(y.view(num_examples, -1), 2, 1) ** 2
        mean_norm = torch.mean((norm / normy) ** (1 / 2))
        return mean_norm

class Conv_Dyn(nn.Module):

    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv_4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.ln = nn.LayerNorm([resolution, resolution], elementwise_affine=True)

    def forward(self, out):
        (u, f, a, diva) = out
        if diva is None:
            diva = self.conv_0(F.tanh(self.conv_1(a)))
        f = self.conv_2(f - diva * u)
        u = u + self.conv_4(f)
        out = (u, f, a, diva)
        return out

class MgRestriction(nn.Module):

    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=2, padding=0, bias=False, padding_mode='zeros'):
        super().__init__()
        self.R_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)

    def forward(self, out):
        (u_old, f_old, a_old, diva_old) = out
        if diva_old is None:
            a_old = self.R_1(a_old)
        u = self.R_2(u_old)
        f = self.R_3(f_old)
        out = (u, f, a_old, diva_old)
        return out

class MG_fem(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, num_iteration=[1, 1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iteration = num_iteration
        self.resolutions = [480, 239, 119, 59, 29, 14, 6]
        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration) - 1):
            if j == 0 or j == 5 or j == 6:
                self.RTlayers.append(nn.ConvTranspose2d(in_channels=in_channels * (j + 2), out_channels=out_channels * (j + 1), kernel_size=4, stride=2, padding=0, bias=False))
            else:
                self.RTlayers.append(nn.ConvTranspose2d(in_channels=in_channels * (j + 2), out_channels=out_channels * (j + 1), kernel_size=3, stride=2, padding=0, bias=False))
        layers = []
        for (l, num_iteration_l) in enumerate(num_iteration):
            for i in range(num_iteration_l):
                layers.append(Conv_Dyn(in_channels=in_channels * (l + 1), out_channels=out_channels * (l + 1), resolution=self.resolutions[l]))
            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            if l < len(num_iteration) - 1:
                layers = [MgRestriction(in_channels=in_channels * (l + 1), out_channels=out_channels * (l + 2))]

    def forward(self, u, f, a, diva_list=[None for _ in range(7)]):
        out_list = [0] * len(self.num_iteration)
        if diva_list[0] is None:
            for l in range(len(self.num_iteration)):
                out = (u, f, a, diva_list[l])
                (u, f, a, diva) = getattr(self, 'layer' + str(l))(out)
                out_list[l] = (u, f, a)
                diva_list[l] = diva
        else:
            for l in range(len(self.num_iteration)):
                out = (u, f, a, diva_list[l])
                (u, f, a, diva) = getattr(self, 'layer' + str(l))(out)
                out_list[l] = (u, f, a)
        for j in range(len(self.num_iteration) - 2, -1, -1):
            (u, f, a) = (out_list[j][0], out_list[j][1], out_list[j][2])
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = (u_post, f, a)
        return (out_list[0][0], out_list[0][1], out_list[0][2], diva_list)

class MgNO(nn.Module):

    def __init__(self, lifting=None, proj=None, dim_input=4, features=12, loss='l2', learning_rate=0.01, step_size=100, gamma=0.5, weight_decay=1e-05, eta_min=0.0005, normalize_param=None, iteration=1):
        super(MgNO, self).__init__()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.iteration = iteration
        (mean_sos, std_sos) = (torch.tensor(1488.3911, dtype=torch.float32), torch.tensor(27.5279, dtype=torch.float32))
        self.register_buffer('mean_sos', mean_sos)
        self.register_buffer('std_sos', std_sos)
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = RRMSE()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'rel_l2':
            self.criterion = LpLoss()
            self.criterion_val = RRMSE()
        if lifting is None:
            self.lifting_1 = nn.Conv2d(2, features, kernel_size=1, bias=False)
            self.lifting_2 = nn.Conv2d(2, features, kernel_size=1, bias=False)
            self.lifting_3 = nn.Conv2d(1, features, kernel_size=1)
        else:
            self.lifting = lifting
        if proj is None:
            self.proj = nn.Conv2d(features, 2, kernel_size=1)
        else:
            self.proj = proj
        self.mgno = nn.ModuleList()
        for l in range(6):
            self.mgno.append(MG_fem(in_channels=features, out_channels=features, num_iteration=[1, 1, 1, 1, 1, 1]))
        self.val_iter = 0

    def forward(self, sos, theta):
        sos = (sos - self.mean_sos) / (self.std_sos * 0.6)
        u = self.lifting_1(theta)
        f = self.lifting_2(theta)
        a = self.lifting_3(sos)
        (u, f, a, diva_list) = self.mgno[0](u, f, a, diva_list=[None for _ in range(6)])
        (u, f, a, diva_list) = self.mgno[1](u, f, a, diva_list=[None for _ in range(6)])
        for _ in range(self.iteration):
            (u, f, a, diva_list) = self.mgno[1](u, f, a, diva_list)
        u = self.proj(u)
        u = theta + u
        return u
