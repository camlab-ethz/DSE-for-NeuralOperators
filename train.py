
"""
This is the main file for training any model on the chosen experiment.
"""

import torch
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import importlib

from _Utilities.Adam import Adam
from _Utilities.utilities import count_params, percentage_difference

import pdb

################################################################
# configs
################################################################
configs = {
    'model':                'fno_smm',                 # Model to train - fno, ffno, ufno, geo_fno, geo_ffno, geo_ufno, fno_smm, ffno_smm, ufno_smm
    'experiment':           'Burgers',               # Burgers, Elasticity, Airfoil, ShearLayer, Humidity
    'device':               torch.device('cuda'),       # Define device for training & inference - GPU/CPU

    ### Data specific parameters
    # 'datapath':             '_Data/Elasticity/',      # Path to data
    'datapath':             '../data/burgers/',      # Path to data
    # 'num_train':            1000,                     # Number of training samples
    # 'num_test':             20,                       # Number of test samples
    'batch_size':           1,                       # Batch size
    # 'epochs':               501,                      # Number of epochs
    # 'test_epochs':          1,                       # How often we print test error during training

    ### Training specific parameters
    # 'learning_rate':        0.005,                    # Learning rate
    # 'scheduler_step':       10,                       # Scheduler step size
    # 'scheduler_gamma':      0.97,                     # Scheduler gamma
    # 'weight_decay':         1e-4,                     # Weight decay
    # 'iphi_loss_reg':        0.0,                      # Regularization parameter for IPHI loss term for the diffeomorphism models
    # 'loss_fn':              'L1',                     # Loss function to use - L1, L2

    ### Saving and loading models
    'save_model':           False,                       # Whether to save the model or not
    'load_model':           False,                      # Whether to load a pretrained model or not, need to specify the model_path then.
    'model_path':           '_Models/model.pt',         # Path to model file if loading model

    
    ### Model specific parameters
    # 'modes1':               12,                       # Number of x-modes to use in the Fourier layer
    # 'modes2':               12,                       # Number of y-modes to use in the Fourier layer
    # 'width':                32,                       # Number of channels in the convolutional layers
    # 'in_channels':          2,                        # Number of channels in input linear layer
    # 'out_channels':         1,                        # Number of channels in output linear layer
    # 'is_mesh':              True,                     # Is it a mesh?
    # 's1':                   40,                       # Number of x-points on latent space GeoFNO grid
    # 's2':                   40,                       # Number of y-points on latent space GeoFNO grid

    ### Specifically for Burgers
    'data_dist':            'conexp',                # Data distribution to use - uniform, conexp, cubic_from_conexp

    ### Specifically for Airfoil
    # 'data_small_domain':    True,                     # Whether to use a small domain or not for specifically the Airfoil experiment

    ### Specifically for Shear Layer
    # 'center_1':         256,                          # Center of top interface
    # 'center_2':         768,                          # Center of bottom interface
    # 'uniform':          100,                          # Number of points uniform along interface
    # 'growth':           1.0,                          # Growth factor, how quickly do points become sparse

    ### Specifically for Humidity
    # 'center_lat':       180,                          # Lattitude center of the nonuniform sampling region
    # 'center_lon':       140,                          # Longitude center of the nonuniform sampling region
}


def train (configs):
    """
    Main training function that will load the configs and find the correct model and experiment to run the training for. The function will return all relevant metrics that are computed during training validation.

    Returns:
        training_times (list): The time in seconds it took to train the model for each epoch.
        train_loss_hist (list): The training loss for each epoch.
        test_loss_hist (list): The test loss for each epoch.
        relative_error_hist (list): The average (over test dataset) relative error for each epoch.
        relative_median_error_hist (list): The median (over test dataset) relative error for each epoch.
    """
    device = configs['device']

    
    ### Load Model
    try:
        ### Regular Grids
        if configs['model'].lower() == 'fno':
            Model = importlib.import_module(configs['experiment']+'.architectures').FNO
        elif configs['model'].lower() == 'ffno':
            Model = importlib.import_module(configs['experiment']+'.architectures').FFNO
        elif configs['model'].lower() == 'ufno':
            Model = importlib.import_module(configs['experiment']+'.architectures').UFNO

        ### Irregular Meshes
        elif configs['model'].lower() == 'geo_fno':
            Model = importlib.import_module(configs['experiment']+'.architectures').Geo_FNO
        elif configs['model'].lower() == 'geo_ffno':
            Model = importlib.import_module(configs['experiment']+'.architectures').Geo_FFNO
        elif configs['model'].lower() == 'geo_ufno':
            Model = importlib.import_module(configs['experiment']+'.architectures').Geo_UFNO

        ### Structured Matrix Method
        elif configs['model'].lower() == 'fno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').FNO_SMM
        elif configs['model'].lower() == 'ffno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').FFNO_SMM
        elif configs['model'].lower() == 'ufno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').UFNO_SMM

        else:
            raise ValueError('Model not recognized.')
        
        # Replace default configs with experiment specific configs if not specified.
        for key in Model.configs:
            configs.setdefault(key, Model.configs[key])
    except Exception as error:
        print(error)
        raise ValueError('Model not compatible with experiment.')
    
    ### Load Dataset
    try:
        getDataloaders = importlib.import_module(configs['experiment']+'.dataset').getDataloaders
    except Exception as error:
        print(error)
        raise ValueError('Experiment not recognized.')
    
    # NOTE: Some configs will have been overwritten by the defaults in Model.configs, hence we print here!
    print(configs)

    ##############
    # data loaders
    ##############
    start_time = time.time()
    print(f'Loading and processing data.')

    train_loader, test_loader = getDataloaders(configs)
    ### TODO TEMPORARY, unlikely good idea to put point dataset into dictionary
    configs['point_data'] = None if not hasattr(train_loader, "point_data") else train_loader.point_data
    configs['denormalizer'] = None if not hasattr(train_loader, "denormalizer") else train_loader.denormalizer
    
    print(f'Processing finished in {time.time()-start_time:.2f}s.')

    
    #######
    # model
    #######
    # initialize model
    
    if configs['load_model']:
        model = torch.load(configs['model_path']).to(device)
    else:
        model = Model(configs).to(device)
    
    # TODO: Trainable parameters will now include IPHI parameters as well.
    print(f"Number of trainable parameters: {count_params(model)}")
    optimizer = Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['scheduler_step'], gamma=configs['scheduler_gamma'])
    # Define the loss function
    if configs['loss_fn'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif configs['loss_fn'] == 'L2':
        loss_fn = torch.nn.MSELoss('mean')
    else:
        raise ValueError('Loss function not recognized.')

    
    ##########
    # training
    ##########
    training_times = []
    train_loss_hist = []
    test_loss_hist = []
    relative_error_hist = []
    relative_median_error_hist = []
    for epoch in range(configs['epochs']):
        start_train = time.time()
        train_loss = 0
        model.train()
        for inputs, targets in train_loader:
            batch_size = targets.shape[0]
            if isinstance(inputs, list):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs)

            # Compute loss separate for complex numbers
            if predictions.is_complex():
                loss = (
                    loss_fn(predictions.real.view(batch_size, -1), targets.real.view(batch_size, -1)) 
                    + loss_fn(predictions.imag.view(batch_size, -1), targets.imag.view(batch_size, -1))
                )
            else:
                loss = loss_fn(predictions.view(batch_size, -1), targets.view(batch_size, -1))

            # For diffeomorphisms, additional loss term:
            if hasattr(model, "model_iphi") and ('iphi_loss_reg' in configs) and (configs['iphi_loss_reg'] > 0):
                samples_x = torch.rand(batch_size, targets.shape[1], 2).cuda() * 3 - 1 # TODO Hardcoded values, check if applies to all
                samples_xi = model.model_iphi(samples_x)
                loss += configs['iphi_loss_reg'] * loss_fn(samples_xi, samples_x)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= configs['num_train']
        stop_train = time.time()
        training_times.append(stop_train-start_train)

        scheduler.step()
        
        ### Only test every test_epochs epochs.
        if epoch % configs['test_epochs'] > 0:
            continue

        ##########
        # testing
        ##########
        with torch.no_grad():
            test_loss = 0
            relative_error = 0
            median_error = torch.zeros(configs['num_test'])

            model.eval()
            for idx, (inputs, targets) in enumerate(test_loader):
                batch_size = targets.shape[0]
                if isinstance(inputs, list):
                    inputs = [x.to(device) for x in inputs]
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device)

                predictions = model(inputs)

                # For different growth factors and sparse data in Humidity, we only consider a small domain.
                if configs['experiment'] == 'Humidity' and configs['growth'] != 1.0:
                    l, r, b, t = test_loader.crop
                    predictions = predictions[:, b:t, l:r,:]
                    targets = targets[:, b:t, l:r,:]

                # Compute loss separate for complex numbers
                if predictions.is_complex():
                    loss = (
                        loss_fn(predictions.real.reshape(batch_size, -1), targets.real.reshape(batch_size, -1)) 
                        + loss_fn(predictions.imag.reshape(batch_size, -1), targets.imag.reshape(batch_size, -1))
                    )
                    # For metrics we only consider REAL parts
                    predictions = predictions.real
                    targets = targets.real
                else:
                    loss = loss_fn(predictions.reshape(batch_size, -1), targets.reshape(batch_size, -1))
                test_loss += loss.item()

                relative_error += percentage_difference(targets.reshape(batch_size, -1), predictions.reshape(batch_size, -1))
                median_error[idx] = percentage_difference(targets.reshape(batch_size, -1), predictions.reshape(batch_size, -1))

        test_loss /= configs['num_test']
        relative_error /= configs['num_test']
        relative_error_hist.append(relative_error)
        relative_median_error_hist.append(torch.median(median_error))


        print(f"Epoch [{epoch:03d}/{configs['epochs']-1}] in {stop_train-start_train:.2f}s with LR {scheduler.get_last_lr()[0]:.2e}: \tTrain loss {train_loss:.4e} \t- Test loss {test_loss:.4e} \t- Test Error {relative_error:.2f}% \t- Median Test Error {torch.median(median_error).item():.4f}%")
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)

        if relative_error == min(relative_error_hist):
            best_model = copy.deepcopy(model)


    lowest_error = min(relative_error_hist)
    # Plot losses
    plt.figure(figsize=(8,6))
    plt.plot([np.log(x) for x in train_loss_hist], label='Train loss')
    plt.plot([np.log(x) for x in test_loss_hist], label='Test loss')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"_Models/loss_history.png")
    plt.close()

    # Save Model
    if configs['save_model']:
        # Remove unnecessary data from models
        best_model.point_data = None
        best_model.denormalizer = None
        print(f"Experiment: {configs['experiment']} \t- Model: {configs['model']} \t- Error: {lowest_error:.4f}%")
        torch.save(best_model, f"_Models/{configs['experiment']}_{configs['model']}.pt")

    return training_times, train_loss_hist, test_loss_hist, relative_error_hist, relative_median_error_hist



if __name__=='__main__':
    ### Set random seed for reproducibility
    seed = 333
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ## Run training for single sample
    train(configs)

    # Run training for multiple models: TODO figure out bug with copying parameters
    # models = ['geo_fno', 'geo_ffno', 'geo_ufno', 'fno_smm', 'ffno_smm', 'ufno_smm']
    # models = ['ffno_smm']
    # for model in models:
        # new_configs = configs
        # new_configs['model'] = model
        # train(new_configs)