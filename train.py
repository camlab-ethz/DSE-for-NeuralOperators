"""
@author: 
"""

"""
This is the main file for training any model on the chosen experiment.
"""

import torch
import numpy as np
import time
import copy

import sys
sys.path.append('../')
from _Utilities.Adam import Adam
from _Utilities.utilities import count_params, percentage_difference

import matplotlib.pyplot as plt
import importlib


################################################################
# configs
################################################################
# TODO: Just give all possible options in comments for the config.
configs = {
    'model':                'fno',                  # Model to train - fno, fno_smm, ffno, ffno_smm, ufno, ufno_smm
    'experiment':           'Elasticity',           # Burgers, Elasticity        
    # 'num_train':            1000,
    # 'num_test':             200,
    # 'batch_size':           20, 
    # 'epochs':               10001,
    # 'test_epochs':          10,

    # Training specific parameters
    # 'learning_rate':        0.0005,
    # 'scheduler_step':       5,
    # 'scheduler_gamma':      0.99,
    # 'weight_decay':         1e-4,                    # Weight decay

    'display_predictions':  False,
    'save_model':           True,
    'load_model':           False,
    'model_path':           '_Models/model.pt',      # Path to model file if loading model
    'min_max_norm':         False,


    'loss_fn':              'L2',                   # Loss function to use - L1, L2
    #'datapath':             '/hdd/mmichelis/VNO_data/elasticity/',  # Path to data

    # Specifically for Burgers
    'data_dist':            'uniform',              # Data distribution to use - uniform, cubic_from_conexp, random
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
    print(configs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### Load Model
    try:
        if configs['model'].lower() == 'fno':
            Model = importlib.import_module(configs['experiment']+'.architectures').FNO
        elif configs['model'].lower() == 'fno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').FNO_SMM
        elif configs['model'].lower() == 'ffno':
            Model = importlib.import_module(configs['experiment']+'.architectures').FFNO
        elif configs['model'].lower() == 'ffno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').FFNO_SMM
        elif configs['model'].lower() == 'ufno':
            Model = importlib.import_module(configs['experiment']+'.architectures').UFNO
        elif configs['model'].lower() == 'ufno_smm':
            Model = importlib.import_module(configs['experiment']+'.architectures').UFNO_SMM
        else:
            raise ValueError('Model not recognized.')
        
        # Replace default configs with experiment specific configs if not specified.
        for key in Model.configs:
            configs.setdefault(key, Model.configs[key])
    except:
        raise ValueError('Model not compatible with experiment.')
    
    ### Load Dataset
    try:
        getDataloaders = importlib.import_module(configs['experiment']+'.dataset').getDataloaders
    except:
        raise ValueError('Experiment not recognized.')
    
    

    ##############
    # data loaders
    ##############
    start_time = time.time()
    print(f'Loading and processing data.')

    train_loader, test_loader = getDataloaders(configs)
    ### TODO TEMPORARY, unlikely good idea to put point dataset into dictionary
    configs['point_data'] = None if not hasattr(train_loader, "point_data") else train_loader.point_data
    
    print(f'Processing finished in {time.time()-start_time:.2f}s.')

    
    #######
    # model
    #######
    # initialize model
    if configs['load_model']:
        model = torch.load(configs['model_path']).to(device)
    else:
        model = Model(configs).to(device)
    
    print(f"Number of trainable parameters: {count_params(model)}")
    optimizer = Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['scheduler_step'], gamma=configs['scheduler_gamma'])
    # Define the loss function
    if configs['loss_fn'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif configs['loss_fn'] == 'L2':
        loss_fn = torch.nn.MSELoss()
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

            loss = loss_fn(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= configs['num_train']
        stop_train = time.time()
        training_times.append(stop_train-start_train)
        
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

                loss = loss_fn(predictions.view(batch_size, -1), targets.view(batch_size, -1))
                test_loss += loss.item()

                relative_error += percentage_difference(targets.view(batch_size, -1), predictions.view(batch_size, -1))
                median_error[idx] = percentage_difference(targets.view(batch_size, -1), predictions.view(batch_size, -1))

        test_loss /= configs['num_test']
        relative_error /= configs['num_test']
        relative_error_hist.append(relative_error)
        relative_median_error_hist.append(torch.median(median_error))

        scheduler.step()

        print(f"Epoch [{epoch:03d}/{configs['epochs']-1}] in {stop_train-start_train:.2f}s with LR {scheduler.get_last_lr()[0]:.2e}: \tTrain loss {train_loss:.4e} \t- Test loss {test_loss:.4e} \t- Test Error {relative_error:.2f}% \t- Median Test Error {torch.median(median_error).item():.2f}%")
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
        print(f"Experiment: {configs['experiment']} \t- Model: {configs['model']} \t- Error: {lowest_error:.2f}%")
        torch.save(best_model, f"_Models/{configs['experiment']}_{configs['model']}.pt")

    return training_times, train_loss_hist, test_loss_hist, relative_error_hist, relative_median_error_hist



if __name__=='__main__':
    ### Set random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ### Run training for single sample
    train(configs)