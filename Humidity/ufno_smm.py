

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from .fno_dse import SpectralConv2d_SMM, VandermondeTransform

################################################################
# UFNO_SMM (SpectralConv2d same as FNO)
################################################################

# the 2D U-Net
class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        padding = (kernel_size - 1) // 2  # Padding size for 'same' convolution
        
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=1)
        self.conv2 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=1)
        self.conv2_1 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=1)
        self.conv3 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate, padding=1)
        self.conv3_1 = self.conv(output_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=1)
        
        self.deconv2 = self.deconv(output_channels, output_channels, padding=1, stride = 2)
        self.deconv1 = self.deconv(output_channels*2, output_channels, padding=1, stride = 2)
        self.deconv0 = self.deconv(output_channels*2, output_channels, padding=1, stride = 2)
    
        self.output_layer = self.output(output_channels*2, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate, padding=padding)
           
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)[...,:31, :33]
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)[...,:65]
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)[...,:123,:]
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


class UFNO_SMM(nn.Module):
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
        super(UFNO_SMM, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.sparse_x, self.sparse_y = configs['point_data']
        self.padding = 0 # pad the domain if input is non-periodic

        
        # Define Structured Matrix Method
        transform = VandermondeTransform(self.sparse_x, self.sparse_y, self.modes1, self.modes2, configs['device'])

        self.fc0 = nn.Linear(18, self.width)
        self.conv0 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv1 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv2 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv3 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

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
        x3 = self.unet2(x)
        x = x1 + x2 + x3 
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.unet3(x)
        x = x1 + x2 + x3 

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]

        gridx = (self.sparse_x - torch.min(self.sparse_x)) / torch.max(self.sparse_x)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])

        gridy = (self.sparse_y - torch.min(self.sparse_y)) / torch.max(self.sparse_y)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



