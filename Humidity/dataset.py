

import netCDF4 as nc
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import os



def getDataloaders (configs):
    """
    Loads the required data for the Humidity experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    # Define which data format to run the experiment on

    ntrain = configs['num_train']                                  
    ntest = configs['num_test']  
    batch_size = configs['batch_size']  

    center_lat = configs['center_lat']
    center_lon = configs['center_lon']
    uniform = configs['uniform']  
    growth = configs['growth']  
    
    load_mod = LoadEarth(configs['datapath'])
    train_in, train_out, test_in, test_out = load_mod.get_data(ntrain, ntest)
    

    # create the needed distribution, x and y positions
    if growth == 1.0:
        y_pos = torch.tensor(load_mod.lon, dtype=torch.float)[:,0]
        x_pos = torch.tensor(load_mod.lat, dtype=torch.float)[0,:]
    else:
        sparsify = MakeSparse2D(train_in.shape[2], train_in.shape[1])
        x_pos, y_pos, nleft, nbelow = sparsify.generate_ce_distribution(growth, growth, center_lat, center_lon, uniform, uniform)
        train_in = sparsify.get_sparse_data(train_in, x_pos, y_pos)
        train_out = sparsify.get_sparse_data(train_out, x_pos, y_pos)
        test_in = sparsify.get_sparse_data(test_in, x_pos, y_pos)
        test_out = sparsify.get_sparse_data(test_out, x_pos, y_pos)

        l = nleft
        r = nleft+uniform
        b = nbelow
        t = nbelow+uniform

        # Testing will always be performed on a smaller region
        test_in = test_in[:, b:t, l:r,:]
        test_out = test_out[:, b:t, l:r,:]


    # create data loaders
    train_loader = DataLoader(TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_in, test_out), batch_size=1, shuffle=False)

    ### Used for irregularly spaced data
    train_loader.point_data = [x_pos, y_pos]    # TODO: Ugly.

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

    def generate_ce_distribution(self, growth_x, growth_y, center_x, center_y, uniform_x, uniform_y):
        con_exp_x, number_left = self.fixed_simple_ce(growth_x, center_x, uniform_x, self.number_points_x)
        con_exp_y, number_bottom = self.fixed_simple_ce(growth_y, center_y, uniform_y, self.number_points_y)

        return con_exp_x.int(), con_exp_y.int(), int(number_left), int(number_bottom)

    def get_sparse_data(self, data, x, y):
        return torch.index_select(torch.index_select(data, 1, y), 2, x)

    def generate_random_points(self, num_points):

        positions = torch.randperm(self.number_points_x*self.number_points_y)[:num_points]

        return positions

    def get_random_cartesian_data(self, data, positions):


        # this is some weird object that seems to iterate automatically every time it is called.
        # data = dataset[0][0][0,:,:].cpu()
        # pdb.set_trace()
        y_mesh, x_mesh = torch.meshgrid(torch.arange(self.number_points_y), torch.arange(self.number_points_x))

        # pass in as a grid, return flattened matrices
        x_flat = torch.reshape(x_mesh, (self.number_points_x*self.number_points_y, 1))
        y_flat = torch.reshape(y_mesh, (self.number_points_x*self.number_points_y, 1))

        data_flat = torch.reshape(data, (data.shape[0],self.number_points_x*self.number_points_y,data.shape[-1])).cpu()

        # plt.tricontourf(x_flat[:,0], y_flat[:,0], data_flat[0,:,0])
        # plt.show()

        x_sparse = torch.index_select(x_flat, 0, positions)
        y_sparse = torch.index_select(y_flat, 0, positions)
        data_sparse = torch.index_select(data_flat, 1, positions)

        # plt.tricontourf(x_sparse[:,0], y_sparse[:,0], data_sparse[0,:,0])
        # plt.show()

        theta = y_sparse / torch.max(y_sparse) * torch.pi # between 0 and pi
        phi = x_sparse / torch.max(x_sparse) * 2 * torch.pi # between 0 and 2 pi

        return  data_sparse, theta, phi


    def random_points_on_sphere(self, n):
        np.random.seed(0)
        # Generate random points in 3D space
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.random.uniform(-1, 1, n)

        # remove all points with radius greater than 1, or else the points in the corners will play a larger role than they should

        # Normalize the points to lie on the unit sphere
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        mask = magnitude <= 1.0
        magnitude_filtered = magnitude[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]

        x /= magnitude_filtered
        y /= magnitude_filtered
        z /= magnitude_filtered

        # Return the points on the sphere
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) + np.pi

        theta = np.floor(theta*361 / np.pi)
        phi = np.floor(phi*576 / (2*np.pi))

        return theta, phi

    def get_random_sphere_data(self, data, num_points):
        theta, phi = self.random_points_on_sphere(num_points)


        data_sparse = data[:,theta,phi,:]


        theta = torch.from_numpy(theta) / 361 * torch.pi
        phi = torch.from_numpy(phi) / 576 * 2 * torch.pi

        return  data_sparse, theta, phi



class LoadEarth:
    def __init__(self, path):
        self.path = path
        self.file_names = os.listdir(path)
        self.names = ['CDH', 'CDQ', 'EFLUX','EVAP','FRCAN','FRCCN', 'FRCLS', 'HLML', 'QSTAR', 'QLML', 'SPEED', 'TAUX', 'TAUY', 'TLML', 'ULML', 'VLML']
        
        ds = nc.Dataset(f"{path}{self.file_names[0]}")
        self.lat, self.lon = np.meshgrid(ds['lon'][:], ds['lat'][:])
        self.y_shape = self.lat.shape[0]
        self.x_shape = self.lat.shape[1]
        
    def get_data(self, num_train, num_test, time_horizon=6):
        # given:    the number of training and testing samples
        # return:   the inputs and outputs of the selected data
        num_samples = (num_train + num_test)

        # inputs are several types of data all concatenated together, outputs are just QLML
        inputs = torch.zeros((num_samples, self.y_shape, self.x_shape, len(self.names)), dtype=torch.float)
        outputs = torch.zeros((num_samples, self.y_shape, self.x_shape, 1), dtype=torch.float)

        time_samples = 24 - time_horizon

        for sample in range(num_train//(24-time_horizon)):
            file = f"{self.path}{self.file_names[sample]}"
            data_set = nc.Dataset(file)

            for index, name in enumerate(self.names):
                inputs[sample*time_samples:(sample+1)*time_samples,:,:, index] = torch.tensor(data_set[name][:time_samples])

            outputs[sample*time_samples:(sample+1)*time_samples,:,:,0] = torch.tensor(data_set['QLML'][time_horizon:])

        for sample in range(num_test//(24-time_horizon)):
            file = f"{self.path}{self.file_names[-(sample+1)]}"
            data_set = nc.Dataset(file)

            for index, name in enumerate(self.names):
                inputs[num_train+sample*time_samples:num_train+(sample+1)*time_samples,:,:, index] = torch.tensor(data_set[name][:time_samples])

            outputs[num_train+sample*time_samples:num_train+(sample+1)*time_samples,:,:,0] = torch.tensor(data_set['QLML'][time_horizon:])

        inputs = self.normalize(inputs)
        outputs = self.normalize(outputs)

        train_in = inputs[:num_train]
        train_out = outputs[:num_train]

        test_in = inputs[-num_test:]
        test_out = outputs[-num_test:]
        
        return  train_in, train_out, test_in, test_out

    def normalize(self, data):
        normalized_data = torch.zeros_like(data)
        for index in range(data.shape[-1]):
            M = torch.max(data[:,:,:,index])
            m = torch.min(data[:,:,:,index])
            normalized_data[:,:,:,index] = (data[:,:,:,index] - m)/(M - m)
        return normalized_data

