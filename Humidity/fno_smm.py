

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



################################################################
# FNO_dse (VandermondeTransform, SpectralConv2d_dse same as ShearLayer)
################################################################

# class for 2-dimensional Fourier transforms on a nonequispaced lattice of data
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


class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d_dse, self).__init__()

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
        x_ft = self.transformer.forward(x.cfloat())

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  2*self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weights1)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = self.transformer.inverse(out_ft).real

        return x
    

class FNO_dse (nn.Module):
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
        super(FNO_dse, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.sparse_x, self.sparse_y = configs['point_data']
        self.padding = 0 # pad the domain if input is non-periodic

        # Define Structured Matrix Method
        transform = VandermondeTransform(self.sparse_x, self.sparse_y, self.modes1, self.modes2, configs['device'])
        
        self.fc0 = nn.Linear(18, self.width)
        self.conv0 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv1 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv2 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, transform)
        self.conv3 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, transform)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)
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
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

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


