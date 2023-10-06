

import torch
import numpy as np

class Dataset (torch.utils.data.Dataset):
    def __init__(self, rr, xy, sigma):
        self.rr = rr
        self.xy = xy
        self.sigma = sigma
        
    def __len__(self):
        return len(self.rr)
    
    def __getitem__(self, idx):
        # Output to compare against is sigma, rest is input to the neural network
        return (self.rr[idx], self.xy[idx]), self.sigma[idx]
    

def getDataloaders (configs):
    """
    Loads the required data for the Elasticity experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    path_rr = configs['datapath']+'/Meshes/Random_UnitCell_rr_10.npy'
    path_XY = configs['datapath']+'/Meshes/Random_UnitCell_XY_10.npy'
    path_Sigma = configs['datapath']+'/Meshes/Random_UnitCell_sigma_10.npy'

    input_rr = np.load(path_rr, allow_pickle=True)
    input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
    input_xy = np.load(path_XY, allow_pickle=True)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)
    input_s = np.load(path_Sigma, allow_pickle=True)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)

    # Only denormalizer needed for models
    min_s = torch.min(input_s)
    max_s = torch.max(input_s)
    def normalizer (sigma):
        return (sigma - min_s) / (max_s - min_s)
    def denormalizer (sigma):
        return sigma * (max_s - min_s) + min_s

    train_rr = input_rr[:configs['num_train']]
    test_rr = input_rr[-configs['num_test']:]
    train_xy = input_xy[:configs['num_train']]
    test_xy = input_xy[-configs['num_test']:]
    train_s = input_s[:configs['num_train']]
    test_s = input_s[-configs['num_test']:]


    train_loader = torch.utils.data.DataLoader(Dataset(train_rr, train_xy, train_s), batch_size=configs['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(test_rr, test_xy, test_s), batch_size=1, shuffle=False)

    train_loader.denormalizer = denormalizer    # TODO UGLY

    return train_loader, test_loader

