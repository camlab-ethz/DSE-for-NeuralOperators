import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from .fno import SpectralConv2d

################################################################
# UFNO (SpectralConv2d same as FNO)
################################################################
class U_net (nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels,  kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
           
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

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class UFNO(nn.Module):
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
        super(UFNO, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
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
        self.unet5r = U_net(self.width, self.width, 3, 0)
        self.unet3i = U_net(self.width, self.width, 3, 0)
        self.unet4i = U_net(self.width, self.width, 3, 0)
        self.unet5i = U_net(self.width, self.width, 3, 0)
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
        x = x1 + x2 + x3 
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv4(x)
        x2 = self.w4r(x.real) + 1j * self.w4i(x.imag)
        x3 = self.unet4r(x.real) + 1j * self.unet4i(x.imag)
        x = x1 + x2 + x3 
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv5(x)
        x2 = self.w5r(x.real) + 1j * self.w5i(x.imag)
        x3 = self.unet5r(x.real) + 1j * self.unet5i(x.imag)
        x = x1 + x2 + x3 

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

