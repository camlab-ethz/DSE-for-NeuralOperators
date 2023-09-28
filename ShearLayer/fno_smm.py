

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



################################################################
# FFNO_SMM
################################################################

# class for 2-dimensional Fourier transforms on a nonequispaced lattice of data
class VFT:
    def __init__ (self, x_positions, y_positions, x_modes, y_modes, device):
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

        return torch.transpose(V_x, 0, 1), torch.conj(V_x), torch.transpose(V_y, 0, 1), torch.conj(V_y)

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
    

class SpectralConv2d_SMM (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d_SMM, self).__init__()

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
    

class FNO_SMM (nn.Module):
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
        super(FNO_SMM, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.sparse_x, self.y_pos = configs['point_data']
        self.padding = 0 # pad the domain if input is non-periodic

        # Define Structured Matrix Method
        transform = VFT(self.sparse_x, self.y_pos, self.modes1, self.modes2, configs['device'])
        
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv1 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv2 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv3 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv4 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv5 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2, transform)
        
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
        gridx = self.sparse_x / torch.max(self.sparse_x)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


