

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from .fno import FNO
from .ufno import UFNO


################################################################
#  2d fourier layer, FFNO
################################################################
class SpectralConv2d_FFNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_FFNO, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight_1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float))
        self.fourier_weight_2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.fft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.modes1] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.modes1],
                torch.view_as_complex(self.fourier_weight_1))

        xy = torch.fft.ifft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.fft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        
        out_ft[:, :, :self.modes1, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.modes1, :],
                torch.view_as_complex(self.fourier_weight_2))

        xx = torch.fft.ifft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        # x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class FFNO(nn.Module):

    # Set a class attribute for the default configs.
    configs = {
        'num_train':            896,
        'num_test':             128,
        'batch_size':           5, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/ShearLayer/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
    }


    def __init__(self, modes1, modes2, width):
        super(FFNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 0 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_FFNO(self.width, self.width, self.modes1, self.modes2)
        
        self.w00r = nn.Conv2d(self.width, self.width, 1)
        self.w10r = nn.Conv2d(self.width, self.width, 1)
        self.w20r = nn.Conv2d(self.width, self.width, 1)
        self.w30r = nn.Conv2d(self.width, self.width, 1)
        self.w40r = nn.Conv2d(self.width, self.width, 1)
        self.w50r = nn.Conv2d(self.width, self.width, 1)        
        self.w01r = nn.Conv2d(self.width, self.width, 1)
        self.w11r = nn.Conv2d(self.width, self.width, 1)
        self.w21r = nn.Conv2d(self.width, self.width, 1)
        self.w31r = nn.Conv2d(self.width, self.width, 1)
        self.w41r = nn.Conv2d(self.width, self.width, 1)
        self.w51r = nn.Conv2d(self.width, self.width, 1)
        
        self.w00i = nn.Conv2d(self.width, self.width, 1)
        self.w10i = nn.Conv2d(self.width, self.width, 1)
        self.w20i = nn.Conv2d(self.width, self.width, 1)
        self.w30i = nn.Conv2d(self.width, self.width, 1)
        self.w40i = nn.Conv2d(self.width, self.width, 1)
        self.w50i = nn.Conv2d(self.width, self.width, 1)
        self.w01i = nn.Conv2d(self.width, self.width, 1)
        self.w11i = nn.Conv2d(self.width, self.width, 1)
        self.w21i = nn.Conv2d(self.width, self.width, 1)
        self.w31i = nn.Conv2d(self.width, self.width, 1)
        self.w41i = nn.Conv2d(self.width, self.width, 1)
        self.w51i = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128).to(torch.cfloat)
        self.fc2 = nn.Linear(128, 1).to(torch.cfloat)


    def forward(self, x):
        # x is [batch, T, x, y]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w00r(x.real) + 1j * self.w00i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w01r(x3.real) + 1j * self.w01i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv1(x)
        x2 = self.w10r(x.real) + 1j * self.w10i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w11r(x3.real) + 1j * self.w11i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv2(x)
        x2 = self.w20r(x.real) + 1j * self.w20i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w21r(x3.real) + 1j * self.w21i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv3(x)
        x2 = self.w30r(x.real) + 1j * self.w30i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w31r(x3.real) + 1j * self.w31i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv4(x)
        x2 = self.w40r(x.real) + 1j * self.w40i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w41r(x3.real) + 1j * self.w41i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv5(x)
        x2 = self.w50r(x.real) + 1j * self.w50i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w51r(x3.real) + 1j * self.w51i(x3.imag)
        x = x4 + x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        # gridx = x_pos / torch.max(x_pos)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        # gridy = y_pos / torch.max(y_pos)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
#  2d fourier layer, FNO Structured Matrix
################################################################
class SpectralConv2d_SMM(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d_SMM, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.transformer = transformer

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.transformer.forward(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  2*self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weights1)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = self.transformer.inverse(out_ft)

        return x
    

class FNO_SMM(nn.Module):

    # Set a class attribute for the default configs.
    configs = {
        'num_train':            896,
        'num_test':             128,
        'batch_size':           5, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/ShearLayer/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
    }

    def __init__(self, modes1, modes2, width, transformer, sparse_x):
        super(FNO_SMM, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.x = sparse_x
        self.padding = 0 # pad the domain if input is non-periodic
        
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv1 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv2 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv3 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv4 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv5 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        
        self.w0r = nn.Conv2d(self.width, self.width, 1)
        self.w1r = nn.Conv2d(self.width, self.width, 1)
        self.w2r = nn.Conv2d(self.width, self.width, 1)
        self.w3r = nn.Conv2d(self.width, self.width, 1)
        self.w4r = nn.Conv2d(self.width, self.width, 1)
        self.w5r = nn.Conv2d(self.width, self.width, 1)
        
        self.w0i = nn.Conv2d(self.width, self.width, 1)
        self.w1i = nn.Conv2d(self.width, self.width, 1)
        self.w2i = nn.Conv2d(self.width, self.width, 1)
        self.w3i = nn.Conv2d(self.width, self.width, 1)
        self.w4i = nn.Conv2d(self.width, self.width, 1)
        self.w5i = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128).to(torch.cfloat)
        self.fc2 = nn.Linear(128, 1).to(torch.cfloat)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0r(x.real) + 1j * self.w0i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv1(x)
        x2 = self.w1r(x.real) + 1j * self.w1i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv2(x)
        x2 = self.w2r(x.real) + 1j * self.w2i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv3(x)
        x2 = self.w3r(x.real) + 1j * self.w3i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv4(x)
        x2 = self.w4r(x.real) + 1j * self.w4i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv5(x)
        x2 = self.w5r(x.real) + 1j * self.w5i(x.imag)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        gridx = self.x / torch.max(self.x)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
#  2d fourier layer, FFNO Structured Matrix
################################################################
class SpectralConv2d_FFNO_SMM(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d_FFNO_SMM, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight_1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float))
        self.fourier_weight_2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float))

        self.transformer = transformer

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = self.transformer.forward_y(x.permute(0,1,3,2).cfloat())
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, N, self.modes1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.modes1] = torch.einsum("bixy,ioy->boxy", x_fty[:, :, :, :self.modes1], torch.view_as_complex(self.fourier_weight_1))

        xy = self.transformer.inverse_y(out_ft).permute(0,1,3,2).real
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = self.transformer.forward_x(x.cfloat())
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M, self.modes1)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :, :self.modes1] = torch.einsum("bixy,ioy->boxy", x_ftx[:, :, :, :self.modes1], torch.view_as_complex(self.fourier_weight_2))

        xx = self.transformer.inverse_x(out_ft).real
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        # x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class FFNO_SMM(nn.Module):

    # Set a class attribute for the default configs.
    configs = {
        'num_train':            896,
        'num_test':             128,
        'batch_size':           5, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/ShearLayer/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
    }

    def __init__(self, modes1, modes2, width, transformer, sparse_x, sparse_y):
        super(FFNO_SMM, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 0 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv1 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv2 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv3 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv4 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv5 = SpectralConv2d_FFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        
        self.w00r = nn.Conv2d(self.width, self.width, 1)
        self.w10r = nn.Conv2d(self.width, self.width, 1)
        self.w20r = nn.Conv2d(self.width, self.width, 1)
        self.w30r = nn.Conv2d(self.width, self.width, 1)
        self.w40r = nn.Conv2d(self.width, self.width, 1)
        self.w50r = nn.Conv2d(self.width, self.width, 1)        
        self.w01r = nn.Conv2d(self.width, self.width, 1)
        self.w11r = nn.Conv2d(self.width, self.width, 1)
        self.w21r = nn.Conv2d(self.width, self.width, 1)
        self.w31r = nn.Conv2d(self.width, self.width, 1)
        self.w41r = nn.Conv2d(self.width, self.width, 1)
        self.w51r = nn.Conv2d(self.width, self.width, 1)
        
        self.w00i = nn.Conv2d(self.width, self.width, 1)
        self.w10i = nn.Conv2d(self.width, self.width, 1)
        self.w20i = nn.Conv2d(self.width, self.width, 1)
        self.w30i = nn.Conv2d(self.width, self.width, 1)
        self.w40i = nn.Conv2d(self.width, self.width, 1)
        self.w50i = nn.Conv2d(self.width, self.width, 1)
        self.w01i = nn.Conv2d(self.width, self.width, 1)
        self.w11i = nn.Conv2d(self.width, self.width, 1)
        self.w21i = nn.Conv2d(self.width, self.width, 1)
        self.w31i = nn.Conv2d(self.width, self.width, 1)
        self.w41i = nn.Conv2d(self.width, self.width, 1)
        self.w51i = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128).to(torch.cfloat)
        self.fc2 = nn.Linear(128, 1).to(torch.cfloat)


    def forward(self, x):
        # x is [batch, T, x, y]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w00r(x.real) + 1j * self.w00i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w01r(x3.real) + 1j * self.w01i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv1(x)
        x2 = self.w10r(x.real) + 1j * self.w10i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w11r(x3.real) + 1j * self.w11i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv2(x)
        x2 = self.w20r(x.real) + 1j * self.w20i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w21r(x3.real) + 1j * self.w21i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv3(x)
        x2 = self.w30r(x.real) + 1j * self.w30i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w31r(x3.real) + 1j * self.w31i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv4(x)
        x2 = self.w40r(x.real) + 1j * self.w40i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w41r(x3.real) + 1j * self.w41i(x3.imag)
        x = F.gelu(x4.real) + 1j * F.gelu(x4.imag) + x
        
        
        x1 = self.conv5(x)
        x2 = self.w50r(x.real) + 1j * self.w50i(x.imag)
        x3 = F.gelu(x2.real) + 1j * F.gelu(x2.imag)
        x4 = self.w51r(x3.real) + 1j * self.w51i(x3.imag)
        x = x4 + x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        # gridx = x_pos / torch.max(x_pos)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        # gridy = y_pos / torch.max(y_pos)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
#  2d fourier layer, UFNO Structured Matrix
################################################################
class SpectralConv2d_UFNO_SMM(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d_UFNO_SMM, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.transformer = transformer

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.transformer.forward(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  2*self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weights1)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = self.transformer.inverse(out_ft)

        return x

class UFNO_SMM(nn.Module):

    # Set a class attribute for the default configs.
    configs = {
        'num_train':            896,
        'num_test':             128,
        'batch_size':           5, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/ShearLayer/",  # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-5,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes':                16,                     # Number of modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
    }
    
    def __init__(self, modes1, modes2, width, transformer, sparse_x):
        super(UFNO_SMM, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.x = sparse_x
        self.padding = 0 # pad the domain if input is non-periodic

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        
        self.conv0 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv1 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv2 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv3 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv4 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv5 = SpectralConv2d_UFNO_SMM(self.width, self.width, self.modes1, self.modes2, transformer)
        
        self.w0r = nn.Conv2d(self.width, self.width, 1)
        self.w1r = nn.Conv2d(self.width, self.width, 1)
        self.w2r = nn.Conv2d(self.width, self.width, 1)
        self.w3r = nn.Conv2d(self.width, self.width, 1)
        self.w4r = nn.Conv2d(self.width, self.width, 1)
        self.w5r = nn.Conv2d(self.width, self.width, 1)
        
        self.w0i = nn.Conv2d(self.width, self.width, 1)
        self.w1i = nn.Conv2d(self.width, self.width, 1)
        self.w2i = nn.Conv2d(self.width, self.width, 1)
        self.w3i = nn.Conv2d(self.width, self.width, 1)
        self.w4i = nn.Conv2d(self.width, self.width, 1)
        self.w5i = nn.Conv2d(self.width, self.width, 1)
        
        self.unet3r = U_net(self.width, self.width, 3, 0)
        self.unet4r = U_net(self.width, self.width, 3, 0)
        
        self.unet3i = U_net(self.width, self.width, 3, 0)
        self.unet4i = U_net(self.width, self.width, 3, 0)
        
        self.fc1 = nn.Linear(self.width, 128).to(torch.cfloat)
        self.fc2 = nn.Linear(128, 1).to(torch.cfloat)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0r(x.real) + 1j * self.w0i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv1(x)
        x2 = self.w1r(x.real) + 1j * self.w1i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv2(x)
        x2 = self.w2r(x.real) + 1j * self.w2i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv3(x)
        x2 = self.w3r(x.real) + 1j * self.w3i(x.imag)
        x3 = self.unet3r(x.real) + 1j * self.unet3i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv4(x)
        x2 = self.w4r(x.real) + 1j * self.w4i(x.imag)
        x3 = self.unet4r(x.real) + 1j * self.unet4i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv5(x)
        x2 = self.w5r(x.real) + 1j * self.w5i(x.imag)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]

        gridx = torch.linspace(0,1,size_x, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])

        gridy = torch.linspace(0,1,size_y, dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
#  Vandermonde Transform for Structured Matrix Method
################################################################

# class for 2-dimensional Fourier transforms on nonequispaced data, using the adjoint as an approximate inverse
class VandermondeTransform:
    def __init__(self, x_positions, y_positions, x_modes, y_modes, device):
        # scalte between 0 and 2 pi
        x_positions -= torch.min(x_positions)
        y_positions -= torch.min(y_positions)
        self.x_positions = x_positions / (torch.max(x_positions)+1) * 2 * np.pi
        self.y_positions = y_positions / (torch.max(y_positions)+1) * 2 * np.pi

        self.x_modes = x_modes
        self.y_modes = y_modes

        self.device = device

        self.x_l = x_positions.shape[0]
        self.y_l = y_positions.shape[0]

        self.Vxt, self.Vxc, self.Vyt, self.Vyc = self.make_matrix()

    def make_matrix(self):
        # given:    class variables
        # return: the matrices required for the forward and inverse transformations

        V_x = torch.zeros([self.x_modes, self.x_l], dtype=torch.cfloat).to(self.device)
        for row in range(self.x_modes):
             for col in range(self.x_l):
                V_x[row, col] = torch.exp(-1j * row *  self.x_positions[col]) 
        
        V_x = torch.divide(V_x, np.sqrt(self.x_l))


        V_y = torch.zeros([2 * self.y_modes, self.y_l], dtype=torch.cfloat).to(self.device)
        for row in range(self.y_modes):
             for col in range(self.y_l):
                V_y[row, col] = torch.exp(-1j * row *  self.y_positions[col]) 
                V_y[-(row+1), col] = torch.exp(-1j * (self.y_l - row - 1) *  self.y_positions[col]) 
        V_y = torch.divide(V_y, np.sqrt(self.y_l))

        return torch.transpose(V_x, 0, 1), torch.conj(V_x.clone()), torch.transpose(V_y, 0, 1), torch.conj(V_y.clone())

    def forward(self, data):
        # given:    data (in spatial domain)
        # return:   the Fourier transformation of the data (to Fourier domain)

        data_fwd = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxt)
                    , 2, 3)
                , self.Vyt)
                , 2,3)

        return data_fwd
    
    def inverse(self, data):
        # given:    data (in Fourier domain)
        # return:   the inverse Fourier transformation of the data (to spatial domain)
        
        data_inv = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxc),
                    2, 3),
                self.Vyc),
                2, 3)
        
        return data_inv

    def forward_x(self, data):
        # given:    data in spatial domain
        # return:   the forward transformation just along the x-axis, for FFNO

        data_fwd = torch.matmul(data, self.Vxt)

        return data_fwd

    def inverse_x(self, data):
        # given:    data (in Fourier domain)
        # return:   the inverse Fourier transformation just along x-axis
        
        data_inv = torch.matmul(data, self.Vxc)
        
        return data_inv

    def forward_y(self, data):
        # given:    data in spatial domain
        # return:   the forward transformation just along the x-axis, for FFNO
        
        data_fwd = torch.matmul(data, self.Vyt)

        return data_fwd

    def inverse_y(self, data):
        # given:    data (in Fourier domain)
        # return:   the inverse Fourier transformation just along x-axis
        
        data_inv = torch.matmul(data, self.Vyc[:self.y_modes,:])
        
        return data_inv
