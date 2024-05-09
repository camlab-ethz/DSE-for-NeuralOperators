
import torch
import torch.nn as nn
import torch.nn.functional as F



################################################################
# FFNO_dse (FVFT, SpectralConv2d_dse, FFNO_dse same as Elasticity)
################################################################

# class for fully nonequispaced 2d points, using the F-FNO approach
class FVFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions) + 1)
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions) + 1)
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Y_ = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float().cuda()


        self.V_fwd_X, self.V_inv_X, self.V_fwd_Y, self.V_inv_Y = self.make_matrix()

    def make_matrix(self):
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:])
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]))
        forward_mat_X = torch.exp(-1j* (X_mat))
        forward_mat_Y = torch.exp(-1j* (Y_mat))

        inverse_mat_X = torch.conj(forward_mat_X).permute(0,2,1)
        inverse_mat_Y = torch.conj(forward_mat_Y).permute(0,2,1)

        return forward_mat_X, inverse_mat_X, forward_mat_Y, inverse_mat_Y

    def forward(self, data):
        fwd_X = torch.bmm(self.V_fwd_X, data)
        fwd_Y = torch.bmm(self.V_fwd_Y, data)
        return fwd_X, fwd_Y

    def inverse(self, data_x, data_y):
        inv_X = torch.bmm(self.V_inv_X, data_x)
        inv_Y = torch.bmm(self.V_inv_Y, data_y)
        return inv_X, inv_Y


class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_dse, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transformer):
        x = x.permute(0, 2, 1)
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft, y_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        y_ft = y_ft.permute(0, 2, 1)

        # Multiply relevant Fourier modes
        out_ft_x = self.compl_mul1d(x_ft, self.weights1)
        out_ft_y = self.compl_mul1d(y_ft, self.weights1)

        #Return to physical space
        out_ft_x = out_ft_x.permute(0, 2, 1)
        out_ft_y = out_ft_y.permute(0, 2, 1)

        x, y = transformer.inverse(out_ft_x, out_ft_y) # x [4, 20, 512, 512]
        x = x+y
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real
    

class FFNO_dse (nn.Module):
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
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               15,                     # Number of x-modes to use in the Fourier layer
        'modes2':               15,                     # Number of y-modes to use in the Fourier layer
        'width':                64,                     # Number of channels in the convolutional layers
    }
    def __init__ (self, configs):
        super(FFNO_dse, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 2 # pad the domain if input is non-periodic

        # Predictions are normalized, we need the output denormalized
        self.denormalizer = configs['denormalizer']


        self.fc0 = nn.Linear(2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.w01 = nn.Conv1d(self.width, self.width, 1)
        self.w02 = nn.Conv1d(self.width, self.width, 1)
        self.w11 = nn.Conv1d(self.width, self.width, 1)
        self.w12 = nn.Conv1d(self.width, self.width, 1)
        self.w21 = nn.Conv1d(self.width, self.width, 1)
        self.w22 = nn.Conv1d(self.width, self.width, 1)
        self.w31 = nn.Conv1d(self.width, self.width, 1)
        self.w32 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward (self, x):
        transform = FVFT(x[:,:,0], x[:,:,1], self.modes1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, transform)
        x2 = self.w01(x1)
        x3 = F.gelu(x2)
        x4 = self.w02(x3)
        x = F.gelu(x4) + x

        x1 = self.conv0(x, transform)
        x2 = self.w11(x1)
        x3 = F.gelu(x2)
        x4 = self.w12(x3)
        x = F.gelu(x4) + x

        x1 = self.conv0(x, transform)
        x2 = self.w21(x1)
        x3 = F.gelu(x2)
        x4 = self.w22(x3)
        x = F.gelu(x4) + x

        x1 = self.conv0(x, transform)
        x2 = self.w31(x1)
        x3 = F.gelu(x2)
        x4 = self.w32(x3)
        x = F.gelu(x4) + x

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x