# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

'''
This code is provided by torch-harmonics. 
It is slightly modified to run the SFNO over data that is sampled at random points and then interpolated through a scheme back up 
to the original grid size. Look in MakeSparse2D.torch_interpolate_to_grid() to change the interpolation scheme.
'''
import os
import time

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch_harmonics.examples.sfno import PdeDataset
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO


from scipy.special import lpmv
from scipy.interpolate import Rbf
from torchrbf import RBFInterpolator
import pdb

# wandb logging
# import wandb
# wandb.login()

def l2loss_sphere(solver, prd, tar, relative=False, squared=False):

    loss = solver.integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
    
    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=False):
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def spectral_loss_sphere(solver, prd, tar, relative=False, squared=False):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls*(ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    coeffs = spectral_weights * coeffs
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_coeffs = spectral_weights * tar_coeffs
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def h1loss_sphere(solver, prd, tar, relative=False, squared=False):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls*(ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    h1_coeffs = spectral_weights * coeffs
    h1_norm2 = h1_coeffs[..., :, 0] + 2 * torch.sum(h1_coeffs[..., :, 1:], dim=-1)
    l2_norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    h1_loss = torch.sum(h1_norm2, dim=(-1,-2))
    l2_loss = torch.sum(l2_norm2, dim=(-1,-2))

     # strictly speaking this is not exactly h1 loss 
    if not squared:
        loss = torch.sqrt(h1_loss) + torch.sqrt(l2_loss)
    else:
        loss = h1_loss + l2_loss

    if relative:
        raise NotImplementedError("Relative H1 loss not implemented")

    loss = loss.mean()


    return loss

def fluct_l2loss_sphere(solver, prd, tar, inp, relative=False, polar_opt=0):
    # compute the weighting factor first
    fluct = solver.integrate_grid((tar - inp)**2, dimensionless=True, polar_opt=polar_opt)
    weight = fluct / torch.sum(fluct, dim=-1, keepdim=True)
    # weight = weight.reshape(*weight.shape, 1, 1)
    
    loss = weight * solver.integrate_grid((prd - tar)**2, dimensionless=True, polar_opt=polar_opt)
    if relative:
        loss = loss / (weight * solver.integrate_grid(tar**2, dimensionless=True, polar_opt=polar_opt))
    loss = torch.mean(loss)
    return loss

l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss(reduction="mean")

class MakeSparse2D:
    # this class handles sparse distributions for 2d and the sphere projected to a cartesian grid
    def __init__(self, number_points_x, number_points_y):
        # the data must be equispaced
        self.number_points_x = number_points_x
        self.number_points_y = number_points_y

    def random_points_on_sphere(self, n):
        np.random.seed(0)
        # Generate random points in 3D space
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.random.uniform(-1, 1, n)

        # remove all points with radius greater than 1 (slightly less than half of all points)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        mask = magnitude <= 1.0
        magnitude_filtered = magnitude[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # Normalize the points to lie on the unit sphere
        x /= magnitude_filtered
        y /= magnitude_filtered
        z /= magnitude_filtered

        # Return the points on the sphere
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) + np.pi

        theta = np.floor(theta*self.number_points_y / np.pi)
        phi = np.floor(phi*self.number_points_x / (2*np.pi))

        # remove duplicate points (there are about 2% duplicates)
        # Combine phi and theta into a 2D array
        positions = np.column_stack((phi, theta))

        # Remove duplicate positions
        unique_positions = np.unique(positions, axis=0)

        # Extract the cleaned phi and theta vectors
        phi_index = unique_positions[:, 0]
        theta_index = unique_positions[:, 1]

        phi_angle = torch.from_numpy(phi_index) / self.number_points_x * 2 * torch.pi
        theta_angle = torch.from_numpy(theta_index) / self.number_points_y * torch.pi

        self.theta_index = theta_index
        self.phi_index = phi_index

        return theta_index, phi_index, theta_angle.to(torch.float), phi_angle.to(torch.float)

    def get_random_sphere_data(self, data, theta, phi):

        data_sparse = data[:,:,theta,phi]

        return  data_sparse
    
    # Method to interpolate data to a regular grid
    def interpolate_to_grid(self, data):
        # Create an Rbf interpolator

        phi_grid, theta_grid = np.mgrid[0:256,0:512]
        # Interpolate the data onto the regular grid
        interpolated_data = torch.zeros(data.shape[0], data.shape[1], 256, 512, dtype=torch.float)
        for batch in range(data.shape[0]):
            for channel in range(data.shape[1]):
                rbf = Rbf(self.theta_index,self.phi_index, data[batch,channel,:].cpu(), function='linear')
                interpolated_data[batch, channel,:,:] = torch.from_numpy(rbf(phi_grid, theta_grid))

        return interpolated_data    
        
    def torch_interpolate_to_grid(self, data):
        # Create an Rbf interpolator
        data_coords = torch.stack((torch.from_numpy(self.theta_index), torch.from_numpy(self.phi_index)), dim=-1).to(torch.float).cuda()

        # Query coordinates
        x = torch.arange(256)
        y = torch.arange(512)
        grid_points = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack(grid_points, dim=-1).reshape(-1, 2).to(torch.float).cuda()


        # Interpolate the data onto the regular grid
        interpolated_data = torch.zeros(data.shape[0], data.shape[1], 256, 512, dtype=torch.float)
        for batch in range(data.shape[0]):
            rbf = RBFInterpolator(data_coords, data.permute(0,2,1)[batch,:,:], kernel='gaussian', epsilon=0.1, device='cuda:0')
            interpolated_data[batch, :,:,:] = rbf(grid_points).reshape(256, 512, 3).permute(2,0,1)

        return interpolated_data



def l1_rel_error(truth, test):
    batch_size = truth.shape[0]
    difference = torch.zeros(batch_size)
    for batch in range(batch_size):
        difference[batch] = torch.mean(torch.abs(truth[batch] - test[batch]))/(torch.mean(torch.abs(truth[batch]))).item() * 100
    return difference

def main(train=True, load_checkpoint=False, enable_amp=False):
    
    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1*3600
    dt_solver = 150
    nsteps = dt//dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(256, 512), device=device, normalize=True)
    # select_random_points(dataset)
    # There is still an issue with parallel dataloading. Do NOT use it at the moment     
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)
    solver = dataset.solver.to(device)

    nlat = dataset.nlat
    nlon = dataset.nlon


        # training function
    def train_sfno_model(model, dataloader, optimizer, gscaler, sparsify, theta, phi, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=64, loss_fn='l2'):
        minimum_median = 1000
        train_start = time.time()
        
        for epoch in range(nepochs):

            # time each epoch
            epoch_start = time.time()

            dataloader.dataset.set_initial_condition('random')
            dataloader.dataset.set_num_examples(num_examples)

            # do the training
            acc_loss = 0
            model.train()

            for inp, tar in dataloader:
                # inp shape [batchsize, 3, 256, 512]
                with amp.autocast(enabled=enable_amp):

                    inp_rand = sparsify.get_random_sphere_data(inp, theta, phi)
                    tar_rand = sparsify.get_random_sphere_data(tar, theta, phi)

                    inp = sparsify.torch_interpolate_to_grid(inp_rand).cuda()
                    tar = sparsify.torch_interpolate_to_grid(tar_rand).cuda()


                    prd = model(inp)
                    for _ in range(nfuture):
                        prd = model(prd)

                    if loss_fn == 'l2':
                        loss = l2loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'h1':
                        loss = h1loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'spectral':
                        loss = spectral_loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'fluct':
                        loss = fluct_l2loss_sphere(solver, prd, tar, inp, relative=True)
                    elif loss_fn == 'l1':
                        loss = l1_loss(prd, tar)
                    else:
                        raise NotImplementedError(f'Unknown loss function {loss_fn}')

                acc_loss += loss.item() * inp.size(0)

                optimizer.zero_grad(set_to_none=True)
                # gscaler.scale(loss).backward()
                gscaler.scale(loss).backward()
                gscaler.step(optimizer)
                gscaler.update()

            acc_loss = acc_loss / len(dataloader.dataset)

            epoch_time = time.time() - epoch_start

            dataloader.dataset.set_initial_condition('random')
            dataloader.dataset.set_num_examples(num_valid)

            # perform validation
            valid_loss = 0
            model.eval()
            errors = torch.zeros((num_valid))
            index = 0
            with torch.no_grad():
                for inp, tar in dataloader:
                    # inp_og = inp
                    inp_rand = sparsify.get_random_sphere_data(inp, theta, phi)
                    tar_rand = sparsify.get_random_sphere_data(tar, theta, phi)

                    inp = sparsify.torch_interpolate_to_grid(inp_rand).cuda()
                    

                    prd = model(inp)
                    prd_rand = sparsify.get_random_sphere_data(prd, theta, phi)
                    for _ in range(nfuture):
                        prd = model(prd)
                    
                    if loss_fn == 'l2':
                        loss = l2loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'h1':
                        loss = h1loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'spectral':
                        loss = spectral_loss_sphere(solver, prd, tar, relative=False)
                    elif loss_fn == 'fluct':
                        loss = fluct_l2loss_sphere(solver, prd, tar, inp, relative=True)
                    elif loss_fn == 'l1':
                        loss = l1_loss(prd_rand, tar_rand)
                    else:
                        raise NotImplementedError(f'Unknown loss function {loss_fn}')

                    errors[4*index:4*(index+1)] = l1_rel_error(tar, prd)
                    # errors[4*index:4*(index+1)] = l1_rel_error(inp.cpu(), inp_og.cpu())
                    index+=1

                    valid_loss += loss.item() * inp.size(0)

            valid_loss = valid_loss / len(dataloader.dataset)

            if scheduler is not None:
                scheduler.step(valid_loss)


            print(f'--------------------------------------------------------------------------------')
            print(f'Epoch {epoch} summary:')
            print(f'time taken: {epoch_time}')
            print(f'accumulated training loss: {acc_loss}')
            print(f'relative validation loss: {valid_loss}')
            print(f'median relative error: {torch.median(errors).item()}')


            if torch.median(errors).item() < minimum_median:
                minimum_median = torch.median(errors).item()
                print(f'*** new minimum median: {torch.median(errors).item()}')
                # torch.save(model, f"../../SMM_models/swe/sfno_swe")

            # if wandb.run is not None:
            #     current_lr = optimizer.param_groups[0]['lr']
            #     wandb.log({"loss": acc_loss, "validation loss": valid_loss, "learning rate": current_lr, "median error": torch.median(errors).item()})


        train_time = time.time() - train_start

        print(f'--------------------------------------------------------------------------------')
        print(f'done. Training took {train_time}.')
        print(f' minimum median: {minimum_median}')
        return valid_loss

    # rolls out the FNO and compares to the classical solver
    def autoregressive_inference(model, dataset, path_root, nsteps, autoreg_steps=10, nskip=1, plot_channel=0, nics=20):

        model.eval()

        losses = np.zeros(nics)
        fno_times = np.zeros(nics)
        nwp_times = np.zeros(nics)

        for iic in range(nics):
            ic = dataset.solver.random_initial_condition(mach=0.2)
            inp_mean = dataset.inp_mean
            inp_var = dataset.inp_var

            prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
            prd = prd.unsqueeze(0)
            uspec = ic.clone()

            # ML model
            start_time = time.time()
            for i in range(1, autoreg_steps+1):
                # evaluate the ML model
                prd = model(prd)

                if iic == nics-1 and nskip > 0 and i % nskip == 0:

                    # do plotting
                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4)
                    plt.savefig(path_root+'_pred_'+str(i//nskip)+'.png')
                    plt.clf()

            fno_times[iic] = time.time() - start_time

            # classical model
            start_time = time.time()
            for i in range(1, autoreg_steps+1):
                
                # advance classical model
                uspec = dataset.solver.timestep(uspec, nsteps)

                if iic == nics-1 and i % nskip == 0 and nskip > 0:
                    ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)

                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(ref[plot_channel], fig, vmax=4, vmin=-4)
                    plt.savefig(path_root+'_truth_'+str(i//nskip)+'.png')
                    plt.clf()

            nwp_times[iic] = time.time() - start_time

            # ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
            ref = dataset.solver.spec2grid(uspec)
            prd = prd * torch.sqrt(inp_var) + inp_mean
            losses[iic] = l2loss_sphere(solver, prd, ref, relative=True).item()
            

        return losses, fno_times, nwp_times

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # prepare sparse points
    sparsify = MakeSparse2D(nlon, nlat)
    num_points = 10000 # yields about 5000 valid points
    theta_index, phi_index, theta, phi = sparsify.random_points_on_sphere(num_points)



    # prepare dicts containing models and corresponding metrics
    models = {}
    metrics = {}


    # SFNO models
    # models['sfno_sc3_layer4_edim256_linear']    = partial(SFNO, spectral_transform='sht', filter_type='linear', img_size=(nlat, nlon),
                                                    #  num_layers=4, scale_factor=3, embed_dim=256, operator_type='vector')
    models['sfno_sc3_layer4_edim256_real']      = partial(SFNO, spectral_transform='sht', filter_type='non-linear', img_size=(nlat, nlon),
                                                     num_layers=4, scale_factor=5, embed_dim=256, complex_activation = 'real', operator_type='diagonal')

    root_path = os.path.dirname(__file__)
    for model_name, model_handle in models.items():
        model = model_handle().to(device)

        metrics[model_name] = {}

        num_params = count_parameters(model)
        print(f'number of trainable params: {num_params}')
        metrics[model_name]['num_params'] = num_params

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(root_path, 'checkpoints/'+model_name)))

        # run the training
        if train:
            # run = wandb.init(project="sfno spherical swe", group=model_name, name=model_name + '_' + str(time.time()), config=model_handle.keywords)

            # optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            gscaler = amp.GradScaler(enabled=enable_amp)

            start_time = time.time()


            print(f'Training {model_name}, single step')
            train_sfno_model(model, dataloader, optimizer, gscaler, sparsify, theta_index, phi_index, scheduler, nepochs=200, loss_fn='l1')


        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses, fno_times, nwp_times = autoregressive_inference(model, dataset, os.path.join(root_path,'paper_figures/'+model_name), nsteps=nsteps, autoreg_steps=10)
            metrics[model_name]['loss_mean'] = np.mean(losses)
            metrics[model_name]['loss_std'] = np.std(losses)
            metrics[model_name]['fno_time_mean'] = np.mean(fno_times)
            metrics[model_name]['fno_time_std'] = np.std(fno_times)
            metrics[model_name]['nwp_time_mean'] = np.mean(nwp_times)
            metrics[model_name]['nwp_time_std'] = np.std(nwp_times)
            if train:
                metrics[model_name]['training_time'] = training_time

    df = pd.DataFrame(metrics)
    df.to_pickle(os.path.join(root_path, 'output_data/metrics.pkl'))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)

    main(train=True, load_checkpoint=False, enable_amp=False)
