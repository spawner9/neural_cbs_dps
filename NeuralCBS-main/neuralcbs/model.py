import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# from torchsummary import summary

class conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class SpatialModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        '''
        inp: b,in,n,n
        out: b,out,n,n
        '''
        super(SpatialModel, self).__init__()
        self.layer1_conv = conv2d_bn(in_channel, 8)
        self.layer2_conv = conv2d_bn(8, 16)
        self.layer3_conv = conv2d_bn(16, 32)
        
        self.layer4_conv = conv2d_bn(32, 16)  
        self.layer5_conv = conv2d_bn(16, 8)  
        self.layer6_conv = nn.Conv2d(8, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.deconv1 = deconv2d_bn(32, 16)  
        self.deconv2 = deconv2d_bn(16, 8)   
    def forward(self, x):
        conv1 = self.layer1_conv(x)# b,8,n,n
        pool1 = F.max_pool2d(conv1, 2)# b,8,n/2,n/2
        conv2 = self.layer2_conv(pool1)# b,16,n/2,n/2
        pool2 = F.max_pool2d(conv2, 2)# b,16,n/4,n/4
        conv3 = self.layer3_conv(pool2)# b,32,n/4,n/4
        convt1 = self.deconv1(conv3)# b,16,n/2,n/2
        concat1 = torch.cat([convt1, conv2], dim=1)# b,32,n/2,n/2
        conv4 = self.layer4_conv(concat1)# b,16,n/2,n/2

        convt2 = self.deconv2(conv4)# b,8,n,n
        concat2 = torch.cat([convt2, conv1], dim=1)# b,16,n,n
        conv5 = self.layer5_conv(concat2)# b,8,n,n
        outp = self.layer6_conv(conv5)# b,out,n,n
        return outp

class SpatialLinearModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 8, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(8)
        # self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, out_channel, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(out_channel)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    

class FCN4chanel(nn.Module):
    def __init__(self, layers: list=[2,5,15,20]):
        super().__init__()
        self.nets = nn.ModuleList([nn.Sequential(nn.Conv2d(i,j,kernel_size=1),
                                                 nn.ReLU()
                                                 ) 
                                   for i,j in zip(layers[:-2],layers[1:-1])
                                  ])
        self.nets.append(nn.Conv2d(layers[-2],layers[-1],kernel_size=1))
        self.nets = nn.Sequential(*self.nets)
    def forward(self, inp):
        return self.nets(inp)

# class FCLayer(nn.Module):
#     """Fully connected layer """
#     def __init__(self, in_feature, out_feature): 
#         super().__init__()
#         self.LinearBlock = nn.Linear(in_feature,out_feature)
#         self.act = nn.ReLU()
#     def forward(self, x):
#         return self.act(self.LinearBlock(x))
# class FCN4chanel(nn.Module):
#     """Simple MLP to code lifting and projection"""
#     def __init__(self, layers = [2, 5, 10, 20]):
#         super().__init__()
#         self.net = nn.ModuleList([FCLayer(in_feature= m, 
#                                           out_feature= n
#                                           ) for m, n in zip(layers[:-2], layers[1:-1])
#                                 ])
#         self.net.append(nn.Linear(in_features= layers[-2], out_features= layers[-1]))

#     def forward(self,x):
#         x = x.permute(0, 2, 3, 1)
#         for module in self.net:
#             x = module(x)
#         x = x.permute(0, 3, 1, 2)
#         return x
class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    def forward(self, x):
        #input: batch,channel,x,y
        #out: batch,channel,x,y
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x))#input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x

class Fourier_layer(nn.Module):
    def __init__(self,  in_,out_, wavenumber1, wavenumber2, is_last = False):
        super(Fourier_layer, self).__init__()
        self.W =  nn.Conv2d(in_, out_, 1)
        self.fourier_conv = fourier_conv_2d(in_, out_ , wavenumber1, wavenumber2)
        if is_last== False: 
            self.act = F.relu
        else: 
            self.act = nn.Identity()
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2) 
    
class FNO(nn.Module):
    def __init__(self, features, wavenumbers=[100,100]):
        super(FNO, self).__init__()
        self.f = nn.ModuleList([Fourier_layer(features,features,
                                              wavenumber1=i,
                                              wavenumber2=i) 
                                              for i in wavenumbers[:-1]])
        self.f.append(Fourier_layer(features,
                                    features,
                                    wavenumber1=wavenumbers[-1],
                                    wavenumber2=wavenumbers[-1],
                                    is_last=True))
    def forward(self, x):
        for f in self.f:
            x = f(x)
        return x
    
class Neural_CBS(nn.Module):
    def __init__(self,
                 lefting=[2,5,10,20],
                 projection=[20,10,5,2],
                 padding=16,
                 spatial_size=480):
        super().__init__()
        # Non-linear mapping of V
        self.sos_model = SpatialModel(in_channel=1,out_channel=2) # (b,1,n,n) --> (b,2,n,n)
        # linear mapping of S
        self.src_model = SpatialLinearModel(in_channel=2,out_channel=2) # (homogenious solution)(b,2,n,n)
        
        self.lefting = FCN4chanel(layers=lefting) # (b,2,n,n) --> (b,f,n,n)
        # Non-linear mapping of G = \mathcal{F}^{-1} \tilde{g}_0 \mathcal{F}
        self.fourier_g = FNO(features=lefting[-1],wavenumbers=[100,100]) # irfft2(g_0*rfft2(...)): (b,f,n+p,n+p) --> (b,f,n+p,n+p)
        self.projection = FCN4chanel(layers=projection) # (b,f,n,n) --> (b,2,n,n)
        self.padding = nn.ZeroPad2d(padding) # (b,2,n,n) --> (b,2,n+p,n+p)
        self.unpadding = nn.ZeroPad2d(-padding)# (b,2,n+p,n+p) --> (b,2,n,n)
        # Non-linear mapping of \gamma
        self.precond = SpatialModel(in_channel=2,out_channel=2)# (b,2,n,n) --> (b,2,n,n)
        
    def forward(self,psi,sos,src):
        '''
        psi: (b,2,n,n)
        sos: (b,1,n,n)
        src: (b,2,n,n)
        \psi_{k+1} = \psi_k - \gamma(\psi_k - G(V\psi_k + S))
        '''
        sos = self.sos_model(sos)# (b,2,n,n)
        src = self.src_model(src)# (b,2,n,n)
        ## tmp = gamma*(psi-ifft2(g_0*fft2(V*psi+S))) ##
        tmp = self.padding(self.lefting(sos*psi + src)) # (b,f,n+p,n+p)
        tmp = self.fourier_g(tmp)# (b,f,n+p,n+p)
        tmp = self.projection(self.unpadding(tmp))# (b,2,n,n)
        tmp = (psi - tmp)
        tmp = self.precond(sos)*tmp
        ##--------------------------------------------##
        psi = psi - tmp
        return psi
    
    def multi_step(self,psi,sos,src,num_step):
        for i in range(num_step):
            psi = self.forward(psi,sos,src)
        return psi
    

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
    
if __name__ == "__main__":
    import time
    x1 = torch.rand(4,2,480,480).to('cpu')
    x2 = torch.rand(4,1,480,480).to('cpu')
    x3 = torch.rand(4,2,480,480).to('cpu')
    x4 = torch.rand(4,10,480,480).to('cpu')
    model = Neural_CBS(lefting=[2,5,10,20],
                    projection=[20,10,5,2],
                    padding=8,
                    spatial_size=480).to('cpu')
    s = time.time()
    print(model(x1,x2,x3).shape)
    e = time.time()
    print(f"time used: {e-s:.4f}s")
    # model = FCN4chanel().to("cuda")
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    num_params = get_parameter_number(model)
    print(f"Total: {num_params['Total']} | Trainable: {num_params['Trainable']}")
    print(f"Total: {num_params['Total']*4/1024**2} MB | Trainable: {num_params['Trainable']*4/1024**2} MB")



