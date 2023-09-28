

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
from einops import rearrange

import copy

################################################################
# GeoFNO
################################################################
class SpectralConv2d (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2


        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u
    

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c
        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)
        # x = x_in

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2
        
        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)
        # Y (batch, channels, N)
        u = u + 0j
        # t1 = default_timer()
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        # t2 = default_timer()
        # print(t2-t1)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)
        # x = x_out

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)
        
        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        # print(f"matrix constructed in {t2-t1} s, transformation in {t3-t2} s")
        Y = Y.real
        return Y


class IPHI (nn.Module):
    """
    inverse phi: x -> xi
    """
    def __init__(self, width=32):
        super(IPHI, self).__init__()

        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 4*self.width)
        self.fc4 = nn.Linear(4*self.width, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.0001,0.0001], device="cuda").reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)


    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


class Geo_FNO (nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .

    input: the solution of the coefficient function and locations (a(x, y), x, y)
    input shape: (batchsize, x=s, y=s, c=3)
    output: the solution 
    output shape: (batchsize, x=s, y=s, c=1)
    """
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             200,
        'batch_size':           20, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Elasticity/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.001,
        'scheduler_step':       50,
        'scheduler_gamma':      0.5,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               12,                     # Number of x-modes to use in the Fourier layer
        'modes2':               12,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
        'in_channels':          2,                      # Number of channels in input linear layer
        'out_channels':         1,                      # Number of channels in output linear layer
        'is_mesh':              True,                     # Is it a mesh?
        's1':                   40,                     # Number of x-points on latent space GeoFNO grid
        's2':                   40,                     # Number of y-points on latent space GeoFNO grid
    }
    def __init__ (self, configs):
        super(Geo_FNO, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.is_mesh = configs['is_mesh']
        self.s1 = configs['s1']
        self.s2 = configs['s2']

        ### Diffeomorphism for GeoFNO iphi
        self.model_iphi = IPHI()    # Will be moved to same device as rest of model

        self.fc0 = nn.Linear(configs['in_channels'], self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, configs['out_channels'])

    def forward (self, x):
        # u (batch, Nx, d) the input value (xy)
        # code (batch, Nx, d) the input features (rr)
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        code, u = x
        x_in, x_out = None, None

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)
        

        u = self.fc0(u)     #[20, 972, 2]
        u = u.permute(0, 2, 1)

        # [20, 32, 40, 40]
        uc1 = self.conv0(u, x_in=x_in, iphi=self.model_iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=self.model_iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3
        #[20, 32, 972]
        u = u.permute(0, 2, 1)
        u = self.fc1(u)     #[20, 972, 128]
        u = F.gelu(u)
        u = self.fc2(u)     #[20, 972, 2]
        return u



    def get_grid (self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



################################################################
# Geo UFNO
################################################################
# Geo UFNO uses the same SpectralConv2d and IPHI as GeoFNO

class U_net (nn.Module):
    # the 2D U-Net
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        padding = (kernel_size - 1) // 2  # Padding size for 'same' convolution
        
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=padding)
        self.conv2 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=padding)
        self.conv2_1 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=padding)
        self.conv3 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=padding)
        self.conv3_1 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=padding)
        
        self.deconv2 = self.deconv(output_channels, output_channels, padding=padding, stride = 2)
        self.deconv1 = self.deconv(output_channels*2, output_channels, padding=padding, stride = 2)
        self.deconv0 = self.deconv(output_channels*2, output_channels, padding=padding, stride = 2)
    
        self.output_layer = self.output(output_channels*2, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=padding)
           
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate, padding):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels, padding, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=stride, padding=padding),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate, padding):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding)


class Geo_UFNO (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             200,
        'batch_size':           20, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Elasticity/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.001,
        'scheduler_step':       50,
        'scheduler_gamma':      0.5,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               12,                     # Number of x-modes to use in the Fourier layer
        'modes2':               12,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
        'in_channels':          2,                      # Number of channels in input linear layer
        'out_channels':         1,                      # Number of channels in output linear layer
        'is_mesh':              True,                     # Is it a mesh?
        's1':                   40,                     # Number of x-points on latent space GeoFNO grid
        's2':                   40,                     # Number of y-points on latent space GeoFNO grid
    }
    def __init__ (self, configs):
        super(Geo_UFNO, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.is_mesh = configs['is_mesh']
        self.s1 = configs['s1']
        self.s2 = configs['s2']

        
        ### Diffeomorphism for GeoFNO iphi
        self.model_iphi = IPHI()    # Will be moved to same device as rest of model

        self.fc0 = nn.Linear(configs['in_channels'], self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, configs['out_channels'])

    def forward(self, x):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        code, u = x
        x_in, x_out = None, None

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)
        

        u = self.fc0(u)     #[20, 972, 2]
        u = u.permute(0, 2, 1)

        # [20, 32, 40, 40]
        uc1 = self.conv0(u, x_in=x_in, iphi=self.model_iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc4 = self.unet2(uc)
        uc = uc1 + uc2 + uc3 + uc4
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc4 = self.unet3(uc)
        uc = uc1 + uc2 + uc3 + uc4
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=self.model_iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3
        #[20, 32, 972]
        u = u.permute(0, 2, 1)
        u = self.fc1(u)     #[20, 972, 128]
        u = F.gelu(u)
        u = self.fc2(u)     #[20, 972, 2]
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



################################################################
# Geo FFNO
################################################################
# Geo FFNO uses the same IPHI as GeoFNO

class WNLinear (nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)


class FeedForward (nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FactorizedSpectralConv2d (nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class SpectralConv2d_FFNO (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32, transform=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        if transform:
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None, transform=True):
        batchsize = u.shape[0]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        if x_in is None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        if transform:
            factor1 = self.compl_mul2d(
                u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = self.compl_mul2d(
                u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        else:
            factor1 = u_ft[:, :, :self.modes1, :self.modes2]
            factor2 = u_ft[:, :, -self.modes1:, :self.modes2]

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1,
                                 s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n_points, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        B = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        # Shift the mesh coords into the right location on the unit square.
        if iphi is None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # x.shape == [B, N, 2]
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)
                         ).reshape(B, N, m1, m2)
        K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)
                         ).reshape(B, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)
                         ).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)
                         ).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class Geo_FFNO (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             200,
        'batch_size':           20, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Elasticity/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               14,                     # Number of x-modes to use in the Fourier layer
        'modes2':               14,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
        'in_channels':          2,                      # Number of channels in input linear layer
        'out_channels':         1,                      # Number of channels in output linear layer
        'n_layers':             4,                      # Number of layers in the network
        'is_mesh':              True,                     # Is it a mesh?
        's1':                   40,                     # Number of x-points on latent space GeoFNO grid
        's2':                   40,                     # Number of y-points on latent space GeoFNO grid
        'share_weight':         False,                  # Share weights across dimensions
    }
    def __init__ (self, configs):
        super(Geo_FFNO, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.is_mesh = configs['is_mesh']
        self.s1 = configs['s1']
        self.s2 = configs['s2']
        self.n_layers = configs['n_layers']

        # Predictions are normalized, we need the output denormalized
        self.denormalizer = configs['denormalizer']

    
        ### Diffeomorphism for GeoFNO iphi
        self.model_iphi = IPHI()    # Will be moved to same device as rest of model

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(configs['in_channels'], self.width)

        self.convs = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.bs = nn.ModuleList([])

        self.fourier_weight = None
        if configs['share_weight']:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(self.width, self.width, self.modes1, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        for i in range(self.n_layers + 1):
            if i == 0:
                conv = SpectralConv2d_FFNO(
                    self.width, self.width, self.modes1, self.modes2, self.s1, self.s2, transform=False)
            elif i == self.n_layers:
                conv = SpectralConv2d_FFNO(
                    self.width, self.width, self.modes1, self.modes2, self.s1, self.s2)
            else:
                conv = FactorizedSpectralConv2d(in_dim=self.width,
                                                out_dim=self.width,
                                                n_modes=self.modes1,
                                                forecast_ff=None,
                                                backcast_ff=None,
                                                fourier_weight=self.fourier_weight,
                                                factor=2,
                                                ff_weight_norm=True,
                                                n_ff_layers=2,
                                                layer_norm=False,
                                                use_fork=False,
                                                dropout=0.0,
                                                mode='full')
            self.convs.append(conv)

        self.bs.append(nn.Conv2d(2, self.width, 1))
        self.bs.append(nn.Conv1d(2, self.width, 1))

        for i in range(self.n_layers - 1):
            w = nn.Conv2d(self.width, self.width, 1)
            self.ws.append(w)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, configs['out_channels'])


    def forward (self, x):
        # u.shape == [batch_size, n_points, 2] are the coords.
        # code.shape == [batch_size, 42] are the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)
        
        code, u = x
        x_in, x_out = None, None

        if self.is_mesh and x_in is None:
            x_in = u
        if self.is_mesh and x_out is None:
            x_out = u

        # grid is like the (x, y) coordinates of a unit square [0, 1]^2
        grid = self.get_grid([u.shape[0], self.s1, self.s2],
                             u.device).permute(0, 3, 1, 2)
        # grid.shape == [batch_size, 2, size_x, size_y] == [20, 2, 40, 40]
        # grid[:, 0, :, :] is the row index (y-coordinate)
        # grid[:, 1, :, :] is the column index (x-coordinate)

        # Projection to higher dimension
        u = self.fc0(u)
        u = u.permute(0, 2, 1)
        # u.shape == [batch_size, hidden_size, n_points]

        uc1 = self.convs[0](u, x_in=x_in, iphi=self.model_iphi,
                            code=code, transform=False)  # [20, 32, 40, 40]
        uc3 = self.bs[0](grid)
        uc = uc1 + uc3

        # uc.shape == [20, 32, 40, 40]
        for i in range(1, self.n_layers):
            uc1 = rearrange(uc, 'b c h w -> b h w c')
            uc1 = self.convs[i](uc1)[0]
            uc1 = rearrange(uc1, 'b h w c -> b c h w')
            # uc2 = self.ws[i-1](uc)
            uc3 = self.bs[0](grid)
            uc = uc + uc1 + uc3

        L = self.n_layers
        u = self.convs[L](uc, x_out=x_out, iphi=self.model_iphi, code=code)
        u3 = self.bs[-1](x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)

        u = self.denormalizer(u) 
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
