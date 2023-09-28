# Structured Matrix Multiplication



## Data and Pre-trained Models
The data has to be downloaded separately from the following location:
**TODOLINK**

Similarly, the trained models that we used in the paper can be found here:
**TODOLINK**


## Training

We formatted the models to be run easily from the `train.py` script. Select the model you wish to run (options `fno`, `ffno`, `ufno`, `geo_fno`, `geo_ffno`, `geo_ufno`, `fno_smm`, `ffno_smm`, `ufno_smm`) and the according experiments (options `Burgers`, `Elasticity`).

The best hyperparameters are chosen by default, when the configs in `train.py` are left empty. These are defined in the separate model architectures of each experiment. You can choose to overwrite these by adding corresponding entries in the configs.


## Hyperparameter Sweep

Running `hyperparameter_sweep.py` with your desired parameters in the configs will grid search for the best architecture.


