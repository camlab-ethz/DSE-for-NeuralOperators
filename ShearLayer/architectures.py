


from .fno import FNO
from .ufno import UFNO
from .ffno import FFNO

from .fno_smm import FNO_SMM
from .ufno_smm import UFNO_SMM
from .ffno_smm import FFNO_SMM




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
