

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Geo UFNO uses the same SpectralConv2d and IPHI as GeoFNO
from .geo_fno import IPHI, SpectralConv2d


################################################################
# Geo UFNO (UNet, SpectralConv2d, IPHI same as Elasticity)
################################################################
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
        'num_train':            1500,
        'num_test':             300,
        'batch_size':           20, 
        'epochs':               501,
        'test_epochs':          10,

        'datapath':             "_Data/Airfoil/",  # Path to data
        'data_small_domain':    True,              # Whether to use a small domain or not for specifically the Airfoil experiment

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
        'n_layers':             4,                      # Number of layers in the network
        'is_mesh':              True,                     # Is it a mesh?
        's1':                   40,                     # Number of x-points on latent space GeoFNO grid
        's2':                   40,                     # Number of y-points on latent space GeoFNO grid
        'share_weight':         False,                  # Share weights across dimensions
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

        u = x
        code = None
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
        
        u = u.permute(0, 2, 1)
        u = self.fc1(u)     
        u = F.gelu(u)
        u = self.fc2(u)     
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

