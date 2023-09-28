

import torch 

from Utilities._utilities import MatReader, normalize


def getDataloaders (configs):
    """
    Loads the required data for the Burgers experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    # Define which data format to run the experiment on
    data_dist = configs['data_dist']

    # Data is of the shape (number of samples, grid size)
    filepath = configs['datapath']+data_dist+'_burgers_data_R10.mat'
    mat_data = MatReader(filepath)
    if data_dist == 'uniform' or data_dist =="cubic_from_conexp":
        x_data = mat_data.read_field('a')[:,:]
        y_data = mat_data.read_field('u')[:,:]
        p_data = torch.arange(x_data.shape[1])
    else:
        x_data = mat_data.read_field('a')[:,:]
        y_data = mat_data.read_field('u')[:,:]
        p_data = mat_data.read_field('loc')[:,:]

    x_data = normalize(x_data)
    y_data = normalize(y_data)


    x_train = x_data[:configs['num_train'],:]
    y_train = y_data[:configs['num_train'],:]
    x_test = x_data[-configs['num_test']:,:]
    y_test = y_data[-configs['num_test']:,:]

    s = x_train.shape[1]

    x_train = x_train.reshape(configs['num_train'],s, 1)
    x_test = x_test.reshape(configs['num_test'],s, 1)
    p_data = p_data.reshape(1, s, 1)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=configs['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)

    ### Used for irregularly spaced data
    train_loader.point_data = p_data    # TODO: Ugly.

    return train_loader, test_loader

