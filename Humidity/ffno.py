
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
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.modes1] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.modes1],
                torch.view_as_complex(self.fourier_weight_1))

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        
        out_ft[:, :, :self.modes1, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.modes1, :],
                torch.view_as_complex(self.fourier_weight_2))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        # x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FFNO (nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            18*50,
        'num_test':             18*10,
        'batch_size':           30, 
        'epochs':               101,
        'test_epochs':          10,

        'datapath':             "_Data/Humidity/",    # Path to data

        # Training specific parameters
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               32,                     # Number of x-modes to use in the Fourier layer
        'modes2':               32,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers

        # Dataset specific parameters
        'center_lat':       180,                        # Lattitude center of the nonuniform sampling region
        'center_lon':       140,                        # Longitude center of the nonuniform sampling region
        'uniform':          100,                        # Width of the nonuniform sampling region
        'growth':           2.0,                        # Growth rate of the nonuniform sampling region
    }
    def __init__(self, configs):
        super(FFNO, self).__init__()
        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(18, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w01 = nn.Conv2d(self.width, self.width, 1)
        self.w11 = nn.Conv2d(self.width, self.width, 1)
        self.w21 = nn.Conv2d(self.width, self.width, 1)
        self.w31 = nn.Conv2d(self.width, self.width, 1)
        self.w02 = nn.Conv2d(self.width, self.width, 1)
        self.w12 = nn.Conv2d(self.width, self.width, 1)
        self.w22 = nn.Conv2d(self.width, self.width, 1)
        self.w32 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # x is [batch, T, x, y]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        # x is [batch, T+2, x, y]
        # x = x.permute(0, 2, 3, 1)
        # x is [batch, x, y, T+2]
        x = self.fc0(x)
        # x is [batch, x, y, modes]
        x = x.permute(0, 3, 1, 2)
        # x is [batch, modes, x, y]
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w01(x1)
        x3 = F.gelu(x2)
        x4 = self.w02(x3)
        x = F.gelu(x4) + x

        x1 = self.conv1(x)
        x2 = self.w11(x1)
        x3 = F.gelu(x2)
        x4 = self.w12(x3)
        x = F.gelu(x4) + x

        x1 = self.conv2(x)
        x2 = self.w21(x1)
        x3 = F.gelu(x2)
        x4 = self.w22(x3)
        x = F.gelu(x4) + x

        x1 = self.conv3(x)
        x2 = self.w31(x1)
        x3 = F.gelu(x2)
        x4 = self.w32(x3)
        x = x4 + x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
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


