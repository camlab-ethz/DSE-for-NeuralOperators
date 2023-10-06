

import torch
import numpy as np

from torch.utils.data import TensorDataset
    

def getDataloaders (configs):
    """
    Loads the required data for the Airfoil experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    path_X = configs['datapath']+'NACA_Cylinder_X.npy'
    path_Y = configs['datapath']+'NACA_Cylinder_Y.npy'
    path_Q = configs['datapath']+'NACA_Cylinder_Q.npy'

    input_x = np.load(path_X, allow_pickle=True)
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = np.load(path_Y, allow_pickle=True)
    input_y = torch.tensor(input_y, dtype=torch.float)
    input_q = np.load(path_Q, allow_pickle=True)
    input_q = torch.tensor(input_q, dtype=torch.float)

    if configs['data_small_domain']:
        width = 45
        depth = 8
        input_x = input_x[:, depth:-depth, :width]
        input_y = input_y[:, depth:-depth, :width]
        input_q = input_q[:, 0, depth:-depth, :width]
    else:
        input_q = input_q[:,0,:,:]

    input_x = torch.flatten(input_x, start_dim=1, end_dim=2)
    input_y = torch.flatten(input_y, start_dim=1, end_dim=2)
    input_q = torch.flatten(input_q, start_dim=1, end_dim=2)
    

    inputs = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), -1)
    outputs = input_q

    train_in = inputs[:configs['num_train']]
    train_out = outputs[:configs['num_train']]
    test_in = inputs[-configs['num_test']:]
    test_out = outputs[-configs['num_test']:]


    train_loader = torch.utils.data.DataLoader(TensorDataset(train_in, train_out), batch_size=configs['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_in, test_out), batch_size=1, shuffle=False)


    return train_loader, test_loader

