

import netCDF4
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np



def getDataloaders (configs):
    """
    Loads the required data for the Shearlayer experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    # Define which data format to run the experiment on

    ntrain = configs['num_train']                                  
    ntest = configs['num_test']  
    batch_size = configs['batch_size']  

    center_points = [configs['center_1']  , configs['center_2']  ]
    uniform = configs['uniform']  
    growth = configs['growth']  

    data_path = configs['datapath']
    
    load_mod = LoadShearflow(ntrain, ntest, file=data_path)
    train_a, train_u, test_a, test_u = load_mod.return_data()

    
    # create the needed distribution, x and y positions
    sparsify = MakeSparse2D(train_a.shape[2], train_a.shape[1])
    if 'smm' in configs['model']:
        train_a, sparse_x = sparsify.shear_distribution(train_a, center_points, growth, uniform)
        train_u = sparsify.shear_distribution(train_u, center_points, growth, uniform)[0]
        test_a = sparsify.shear_distribution(test_a, center_points, growth, uniform)[0]
        test_u = sparsify.shear_distribution(test_u, center_points, growth, uniform)[0]
    else:
        sparse_x = sparsify.shear_distribution(train_a, center_points, growth, uniform)[1]
        sparse_x = sparse_x.int()

    y_pos = torch.arange(1024)

    # create data loaders
    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

    ### Used for irregularly spaced data
    train_loader.point_data = [sparse_x, y_pos]    # TODO: Ugly.

    return train_loader, test_loader


class MakeSparse2D:
    # this class handles sparse distributions for 2d and the sphere projected to a cartesian grid
    def __init__(self, number_points_x, number_points_y):
        # the data must be equispaced
        self.number_points_x = number_points_x
        self.number_points_y = number_points_y

    def fixed_simple_ce(self, growth, center, uniform, number_points):
        if uniform > 1:
            # define the sides of the uniform region
            left_side = center - uniform//2
            right_side = center + uniform//2
            
            
            # define the number of points beyond each side of the uniform region
            number_left = np.floor(left_side**(1/growth))+1
            number_right = np.floor((number_points - right_side)**(1/growth))+1

            # define the positions of points to each side
            points_left = torch.flip(left_side - torch.round(torch.pow(torch.arange(number_left), growth)), [0])
            points_right = right_side + torch.round(torch.pow(torch.arange(number_right), growth))

            uniform_region = torch.arange(left_side+1, right_side, dtype=torch.float)
            con_exp = torch.cat((points_left, uniform_region, points_right))
  

        elif uniform == 0:
            # not necessarily symmetric
            # define the number of points beyond each side of the uniform region
            
            number_left = np.floor(center**(1/growth))+1
            number_right = np.floor((number_points - center)**(1/growth)) + 1

            # define the positions of points to each side
            points_left = torch.flip(center - torch.round(torch.pow(torch.arange(number_left), growth)), [0])
            points_right = center + torch.round(torch.pow(torch.arange(number_right), growth)) - 1
            
            con_exp = torch.cat((points_left, points_right[2:]))
        return con_exp, number_left

    def shear_distribution(self, data, center_points, growth, uniform):
        # the data must have the shape 
        if growth == 1:
            return data, torch.arange(self.number_points_x)
        # center points shold be a list in order of where the highest gradients are
        ce_left = self.fixed_simple_ce(growth, center_points[0], uniform, self.number_points_x//2 - 1)[0]
        ce_right = self.number_points_x - ce_left.flip(0) - 1
        
        ce_left[0] = 0
        ce_right[-1] = self.number_points_x - 1
        
        sparse_distribution = torch.cat((ce_left, ce_right))
        sparse_data = torch.index_select(data, -2, sparse_distribution.int())

        return sparse_data, sparse_distribution


class LoadShearflow():
    def __init__(self, training_samples, testing_samples, file):
        self.in_size = 1024
        self.file = file
        self.ntrain = training_samples
        self.ntest = testing_samples
        training_inputs, training_outputs = self.get_data(training_samples+testing_samples)
        
        training_inputs = training_inputs.permute(0, 2, 3, 1)
        training_outputs = training_outputs.permute(0, 2, 3, 1)

        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        testing_inputs = training_inputs[-testing_samples:] 
        testing_outputs = training_outputs[-testing_samples:] 
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]

        self.testing_inputs = testing_inputs
        self.testing_outputs = testing_outputs
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

        
    def return_data(self):
        return self.training_inputs, self.training_outputs, self.testing_inputs, self.testing_outputs

    def normalize(self, data):
        m = torch.max(data.real)
        M = torch.min(data.real)
        real_data = (data.real - m)/(M - m)

        m = torch.max(data.imag)
        M = torch.min(data.imag)
        imag_data = (data.imag - m)/(M - m)
        
        return real_data + 1j * imag_data

    def get_data(self, n_samples):
        # given:    the total number of samples to get the data from
        # return:   the data in a tensor format

        input_data = np.zeros((n_samples, 1, self.in_size, self.in_size), dtype=np.cfloat)
        output_data = np.zeros((n_samples, 1, self.in_size, self.in_size), dtype=np.cfloat)

        for i in range(self.ntrain):
            # input data
            file_input  = self.file + "sample_" + str(i) + "_time_0.nc" 
            f = netCDF4.Dataset(file_input,'r')
            input_data[i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

            # output data
            file_output = self.file + "sample_" + str(i) + "_time_1.nc" 
            f = netCDF4.Dataset(file_output,'r')
            output_data[i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

        for i in range(self.ntest):
            # input data
            file_input  = self.file + "sample_" + str(896+i) + "_time_0.nc" 
            f = netCDF4.Dataset(file_input,'r')
            input_data[self.ntrain+i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

            # output data
            file_output = self.file + "sample_" + str(896+i) + "_time_1.nc" 
            f = netCDF4.Dataset(file_output,'r')
            output_data[self.ntrain+i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()
            
        return torch.tensor(input_data).type(torch.cfloat), torch.tensor(output_data).type(torch.cfloat)

