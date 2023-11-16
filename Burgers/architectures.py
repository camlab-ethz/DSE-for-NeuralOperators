

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NUFFT imports
# import pycuda.autoinit
# from pycuda.gpuarray import to_gpu
# import pycuda.gpuarray as gpuarray

# from cufinufft import cufinufft

# from Burgers import nufft_utils

# Torch NUFFT imports
import torchkbnufft as tkbn
import matplotlib.pyplot as plt

import pdb

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d (nn.Module):
    def __init__(self, in_channels, out_channels, modes, point_data, transform=None):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.transform = transform

        self.point_data = (point_data / (8192) * 2 * np.pi).cuda()
        self.point_data = self.point_data[:,:,0]

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))


        # for the NUFFT comparison
        self.adjkb_ob = tkbn.KbNufftAdjoint(im_size=(64,)).cuda()
        self.nufft_ob = tkbn.KbNufft(im_size=(64,)).cuda()

        self.toep_ob = tkbn.ToepNufft()


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        if self.transform is not None:
            # FNO SMM
            x_ft = self.transform.forward(x.cfloat())
            out_ft = self.compl_mul1d(x_ft, self.weights)
            x = self.transform.inverse(out_ft).real / x.size(-1) * 2

            # Kaiser-Bessel NUFFT
            # batchsize = x.shape[0]
            # image = x.to(torch.cfloat)
            # omega = self.point_data

            # image_ft = self.nufft_ob(image, omega)
            # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1),  device=x.device, dtype=torch.cfloat)
            # out_ft[:, :, :self.modes] = self.compl_mul1d(image_ft[:, :, :self.modes], self.weights)
            # image_out = self.adjkb_ob(out_ft, omega) / 64
            # x = image_out.real


            # Toeplitz NUFFT
            # batchsize = x.shape[0]
            # image = x.to(torch.cfloat)
            # omega = self.point_data

            # kernel = tkbn.calc_toeplitz_kernel(omega, im_size=(64,))
            # image_out = self.toep_ob(image, kernel)/64
            # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1),  device=x.device, dtype=torch.cfloat)
            # out_ft[:, :, :+self.modes] = self.compl_mul1d(image_out[:, :, :self.modes], self.weights)
            # image_back = self.toep_ob(out_ft, kernel)/(64)
            # x = image_back.real
            
        else:
            # standard FNO
            batchsize = x.shape[0]
            x_ft = torch.fft.rfft(x)
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
            x = torch.fft.irfft(out_ft, n=x.size(-1))


        return x


class FNO (nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the initial condition and location (a(x), x)
    input shape: (batchsize, x=s, c=2)
    output: the solution of a later timestep
    output shape: (batchsize, x=s, c=1)
    """
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             200,
        'batch_size':           50, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Burgers/",  # Path to data
        'data_dist':            'cubic_from_conexp',              # Data distribution to use - uniform, cubic_from_conexp, random

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L2',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                64,                     # Number of channels in the convolutional layers
    }

    def __init__(self, configs):
        super(FNO, self).__init__()

        self.modes = configs['modes']
        self.width = configs['width']
        self.padding = 2 # pad the domain if input is non-periodic
        self.point_data = configs['point_data']

        # Define network
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'])
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'])
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'])
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'])
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(self.point_data, x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, p_data, shape, device):
        batchsize, size_x = shape[0], shape[1]
        # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = p_data / torch.max(p_data)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)



################################################################
#  Vandermonde Transform for Structured Matrix Method
################################################################

# class for 1-dimensional Fourier transforms on nonequispaced data, using the adjoint as an approximate inverse
class VandermondeTransform:
    def __init__(self, positions, modes):
        self.modes = modes
        self.positions = positions / (8192) * 2 * np.pi
        self.l = positions.shape[0]

        self.Vt, self.Vc = self.make_matrix()

    def make_matrix(self):
        V = torch.zeros([self.modes, self.l], dtype=torch.cfloat).cuda()
        for row in range(self.modes):
            V[row,:] = np.exp(-1j * row * self.positions)
        
        V_inv = torch.conj(V.clone())
        V_inv[0,:] = 0.5
        
        return torch.transpose(V, 0, 1), V_inv

    def forward(self, data):
        return torch.matmul(data, self.Vt)

    def inverse(self, data):
        return torch.matmul(data, self.Vc)


class FNO_SMM (nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2.

    Structured Matrix Method implemented in SpectralConv1d by adding a transformer.
    
    input: the solution of the initial condition and location (a(x), x)
    input shape: (batchsize, x=s, c=2)
    output: the solution of a later timestep
    output shape: (batchsize, x=s, c=1)
    """
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1000,
        'num_test':             200,
        'batch_size':           50, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Burgers/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                64,                     # Number of channels in the convolutional layers
    }

    def __init__(self, configs):
        super(FNO_SMM, self).__init__()

        self.modes = configs['modes']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic
        self.point_data = configs['point_data']

        # Define Structured Matrix Method
        transform = VandermondeTransform(self.point_data.squeeze(), self.modes)

        # Define network
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'], transform=transform)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'], transform=transform)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'], transform=transform)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes, configs['point_data'], transform=transform)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(self.point_data, x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, p_data, shape, device):
        batchsize, size_x = shape[0], shape[1]
        # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = p_data / torch.max(p_data)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
