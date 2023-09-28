
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



################################################################
# FFNO 
################################################################
class SpectralConv2d (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

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


class FFNO (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            896,
        'num_test':             128,
        'batch_size':           8, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/ShearLayer/",    # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               20,                     # Number of x-modes to use in the Fourier layer
        'modes2':               20,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers

        # Dataset specific parameters
        'center_1':         256,                        # X-center of the nonuniform sampling region
        'center_2':         768,                        # Y-center of the nonuniform sampling region
        'uniform':          100,                        # Width of the nonuniform sampling region
        'growth':           1.0,                        # Growth rate of the nonuniform sampling region
    }
    def __init__(self, configs):
        super(FFNO, self).__init__()
        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
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


