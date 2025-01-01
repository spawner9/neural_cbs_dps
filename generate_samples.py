import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from scipy.io import loadmat
import os

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

class ConvergentBornSeries_Batch(nn.Module):
    def __init__(self,
                 lamb = 1,
                 sos=None,
                 boundary_width=[8,8],
                 boundary_strength=1,
                 boundary_type='PML3',
                 device = "cuda:0",
                 ):
        super().__init__()
        # Device
        self.device = device
        # # PDE params
        self.lamb = lamb
        self.omega0 = 2*torch.pi / self.lamb 
        
        # Discretization
        self.PPW = 4
        self.pixel_size = self.lamb / self.PPW 
        
        self.sos = nn.Parameter(sos, requires_grad = True).to(device)
        self.N = torch.tensor(sos.size()).to(int)
        
        # Sources Location
        # later as input!
        
        # Boundary Padding
        self.boundary_widths = torch.tensor(boundary_width,dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()
        
        # V(x) Params
        self.k =  self.omega0/self.sos
        self.k0 = torch.sqrt((torch.max(self.k**2) + torch.min(self.k**2))/2).detach()
        
        # CBS Iteration Params
        self.k2_pad = self.set_boundary(self.k)
        self.epsilon = torch.max(torch.abs(self.k2_pad**2 - self.k0**2)).detach()
        self.g0 = 1/(self.px_range**2+self.py_range**2-self.k0**2 - 1.0j*self.epsilon).detach()
        
        self.V = self.k2_pad - self.k0**2 - 1.0j*self.epsilon
        self.gamma = 1.j/self.epsilon*self.V
    
    def forward(self, src_loc_batch, max_iters=5,tol=1e-16, requries_grad = False):
        self.batch_size = src_loc_batch.shape[0]
        self.src_batch = self.set_src_loc(src_loc_batch)
        
        self.u = torch.zeros(self.new_N[0],self.new_N[1]).to(self.device)
        self.u_batch = torch.stack([self.u] * self.batch_size, dim=0) 
        
        # For batch
        self.V = self.V.unsqueeze(0)
        self.g0 = self.g0.unsqueeze(0)
        self.gamma = self.gamma.unsqueeze(0)

        if requries_grad:
            for _ in range(max_iters):
                
                tmp1 = self.V * self.u_batch
                tmp1[:, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src_batch

                tmp2 = (self.gamma) * (self.u_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                self.u_batch = self.u_batch - tmp2
        else:
            with torch.no_grad():
                for _ in range(max_iters):

                    tmp1 = self.V * self.u_batch
                    tmp1[:, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src_batch

                    tmp2 = (self.gamma) * (self.u_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                    self.u_batch = self.u_batch - tmp2

        return self.u_batch[:, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]

    def set_grid(self):
        # Set New Grid for FFT 
        self.new_N = (2**(torch.ceil(torch.log2(self.N + self.boundary_widths)))).to(int).to(self.device) # To improve FFT efficiency

        self.x_range = (torch.arange(self.new_N[0])*self.pixel_size).view(-1,1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1])*self.pixel_size).to(self.device)
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.new_N[0],d=self.pixel_size).view(-1,1).to(self.device)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.new_N[1],d=self.pixel_size)).to(self.device)
        
        # Set Inner Grid Indices
        self.roi_size = torch.tensor(self.sos.size()).to(self.device)
        self.Bl = torch.ceil((self.new_N - self.roi_size)/2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size)/2).to(int)
        self.roi = [[self.Bl[0],self.Bl[0]+self.roi_size[0]],
                    [self.Bl[1],self.Bl[1]+self.roi_size[1]]]
        
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        self.dist = torch.sqrt(self.x**2 + self.y**2).to(self.device)

    def set_boundary(self, k):
        k_pad = F.pad(k.unsqueeze(0), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='replicate').squeeze()
        
        k0 = torch.sqrt(torch.mean(k_pad**2))* self.pixel_size # k0 in 1/pixels
        c = self.boundary_strength*k0**2 / (2*k0)
        f_boundary_curve, _ = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type,
                                                                            r=self.dist,
                                                                            Bmax=self.Bmax,
                                                                            c=c,
                                                                            k0=k0)
        
        k2_pad = k_pad ** 2 + (f_boundary_curve * self.omega0**2).detach()
        return k2_pad

    def compute_leakage_and_boundary_curve(self,boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5': 
            f_boundary_curve = 1 / k0**2 * (c**7 * r**5 * (6.0 + (2.0j * k0 - c) * r)) / \
                (720 + 720 * c * r + 360 * c**2 * r**2 + 120 * c**3 * r**3 + 30 * c**4 * r**4 + 6 * c**5 * r**5 + c**6 * r**6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c**2 * Bmax**2 + 
                                            120 * c**3 * Bmax**3 + 30 * c**4 * Bmax**4 + 
                                            6 * c**5 * Bmax**5 + c**6 * Bmax**6) / 24
        elif boundary_type == 'PML4':  # 4th order smooth
            f_boundary_curve = 1 / k0**2 * (c**6 * r**4 * (5.0 + (2.0j * k0 - c) * r)) / \
                (120 + 120 * c * r + 60 * c**2 * r**2 + 20 * c**3 * r**3 + 5 * c**4 * r**4 + c**5 * r**5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c**2 * Bmax**2 + 
                                            20 * c**3 * Bmax**3 + 5 * c**4 * Bmax**4 + c**5 * Bmax**5) / 24
        elif boundary_type == 'PML3':  # 3rd order smooth
            f_boundary_curve = 1 / k0**2 * (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
                (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)
            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c**2 * Bmax**2 + 
                                            4 * c**3 * Bmax**3 + c**4 * Bmax**4) / 24
        elif boundary_type == 'PML2':  # 2nd order smooth
            f_boundary_curve = 1 / k0**2 * (c**4 * r**2 * (3.0 + (2.0j * k0 - c) * r)) / \
                (6 + 6 * c * r + 3 * c**2 * r**2 + c**3 * r**3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c**2 * Bmax**2 + c**3 * Bmax**3) / 6
        elif boundary_type == 'PML1':  # 1st order smooth
            f_boundary_curve = 1 / k0**2 * (c**3 * r * (2.0 + (2.0j * k0 - c) * r)) / \
                (2.0 + 2.0 * c * r + c**2 * r**2) / k0**2 # (divide by k0^2 to get relative e_r)
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c**2 * Bmax**2) / 2
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        return f_boundary_curve, leakage
    
    def get_u(self):
        return self.u[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]].detach().cpu()
    
    def set_src_loc(self, src_loc_batch = np.array([[30,60],[240,280]])):
        batch_size = src_loc_batch.shape[0]
        src = torch.zeros_like(self.sos).to(self.device)
        src_batch = torch.stack([src] * batch_size, dim=0)
        
        for i in range(batch_size):
            src_loc = src_loc_batch[i]
            src_batch[i][src_loc[0],src_loc[1]] = 1

        src_batch = nn.Parameter(src_batch, requires_grad = False)
        return src_batch
    
if __name__ == "__main__":
    x_pos = loadmat('x_pos.mat')['x_pos256']
    y_pos = loadmat('y_pos.mat')['y_pos256']

    receiver_indices = torch.cat((torch.tensor(x_pos.astype(np.int64)), torch.tensor(y_pos.astype(np.int64))), dim = 1)
    transmitter_indices = torch.cat((torch.tensor(x_pos.astype(np.int64)), torch.tensor(y_pos.astype(np.int64))), dim = 1)

    speed_train_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/train"
    speed_eval_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/eval"

    dobs_300k_train_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_300k/train"
    dobs_300k_eval_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_300k/eval"
    dobs_400k_train_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_400k/train"
    dobs_400k_eval_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_400k/eval"
    dobs_500k_train_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_500k/train"
    dobs_500k_eval_path = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_500k/eval"
    
    for count in range(0,10):
        print(count)

        #speed_train = np.load(os.path.join(speed_train_path, "train_"+str(count+1)+".npy"))
        speed_train = np.load(os.path.join(speed_train_path, "train_"+str(count+1)+".npy"))
        const = 10000
        model_300k = ConvergentBornSeries_Batch(lamb = const/200,
                                sos= torch.tensor(speed_train)/1500,
                                boundary_width= [32,32],
                                boundary_strength= 1,
                                boundary_type= 'PML3',
                                device = device)

        model_400k = ConvergentBornSeries_Batch(lamb = const/266.66,
                                sos= torch.tensor(speed_train)/1500,
                                boundary_width= [32,32],
                                boundary_strength= 1,
                                boundary_type= 'PML3',
                                device = device)

        model_500k = ConvergentBornSeries_Batch(lamb = const/333.33,
                                sos= torch.tensor(speed_train)/1500, 
                                boundary_width= [32,32],
                                boundary_strength= 1,
                                boundary_type= 'PML3',
                                device = device)
        
        u_300k = model_300k(src_loc_batch = transmitter_indices, max_iters=1000, requries_grad= False).detach().cpu().numpy()
        u_400k = model_400k(src_loc_batch = transmitter_indices, max_iters=1000, requries_grad= False).detach().cpu().numpy()
        u_500k = model_500k(src_loc_batch = transmitter_indices, max_iters=1000, requries_grad= False).detach().cpu().numpy()
            
        np.save(os.path.join(dobs_300k_train_path, "train_"+str(count+1)+".npy"), u_300k[:,receiver_indices[:,0], receiver_indices[:,1]])
        np.save(os.path.join(dobs_400k_train_path, "train_"+str(count+1)+".npy"), u_400k[:,receiver_indices[:,0], receiver_indices[:,1]])
        np.save(os.path.join(dobs_500k_train_path, "train_"+str(count+1)+".npy"), u_500k[:,receiver_indices[:,0], receiver_indices[:,1]])
        
        


