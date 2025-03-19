"""
This module implements the VAE training for DW-MRI

Authors:

    Julius Glaser <julius-glaser@gmx.de>
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

from __future__ import division
import h5py
import os
import sys
import yaml
import torch
import numpy as np
import torch.utils.data as data
from yaml import Loader
from copy import deepcopy as dc

#import repo dependencies
from laser.training.sim import dataset
from laser.training.sim import dwi
from laser.training.models.nn import autoencoder as ae
import laser.training.linsub
from laser.training.util_classes import Losses_class as LC
from laser.training.util_classes import Network_parameters as NP

def train(network_parameters: NP, 
          loader_train: data.DataLoader, 
          optim: torch.optim.Optimizer, 
          model: torch.nn.Module, 
          device: torch.device, 
          loss_function: torch.nn.modules.loss, 
          epoch: int, 
          Losses: LC, 
          gamma_x: float, 
          scalingMatrix: np.array)-> LC:
    """
    train function for neural networks, VAE or DAE.

    Args:
        network_parameters (NP): an instance of the NetworkParameters util_classes, 
        loader_train (data.DataLoader): neural network data loader for training data, 
        optim (torch.optim.Optimizer): optimizer, 
        model (torch.nn.Module): model to be trained, 
        device (torch.device): device for training (cpu or cuda), 
        loss_function (torch.nn.modules.loss): loss function for reconstruction loss, 
        epoch (int): current training epoch, 
        Losses (LC): an instance of the Loss_class util_classes, 
        gamma_x (float): KLD optimization according to train loss [1], 
        scalingMatrix (np.array): matrix to scale loss according to b-value

    Returns:
        LC: Losses with added current train loss.

    References:
        [1] Asperti A, Trentin M. Balancing reconstruction error and Kullback-Leibler divergence in Variational Autoencoders.
            CoRR, 2020:abs/2002.07514

    """
    
    # initialize values and set model to train
    model.train()
    scalingMatrix = torch.tensor(scalingMatrix, device=device)
    N_diff = scalingMatrix.shape[0]
    HALF_LOG_TWO_PI = 0.91893
    train_loss = 0.0
    kld_loss = 0.0
    recon_loss = 0.0
    train_loss = 0.0
    mse_train = 0.0

    for batch_idx, (noisy_t, clean_t, _, _) in enumerate(loader_train):

        noisy_t = noisy_t.type(torch.FloatTensor).to(device)
        clean_t = clean_t.type(torch.FloatTensor).to(device)
        optim.zero_grad()

        # denoise inputs, and calculate loss
        if network_parameters.model == 'DAE':
            recon_t = model(noisy_t)
            loss = loss_function(recon_t, clean_t)
        elif network_parameters.model == 'VAE':
            recon_t, mu, logvar = model(noisy_t)
            k = (2*N_diff/network_parameters.latent)**2

            recon_t_scaled = recon_t*scalingMatrix
            clean_t_scaled = clean_t*scalingMatrix

            # auto scaling of kld
            mse = loss_function(recon_t_scaled, clean_t_scaled)
            loggamma_x = np.log(gamma_x)
            gen_loss = torch.sum(torch.square((recon_t_scaled - clean_t_scaled)/gamma_x)/2.0 + loggamma_x + HALF_LOG_TWO_PI)/ network_parameters.batch_size_train
            KLD = ae.loss_function_kld(mu, logvar)/network_parameters.batch_size_train
            loss = gen_loss + k*KLD/network_parameters.batch_size_train
            kld_loss += (k*KLD).item()
            recon_loss += mse.item()
            mse_train += mse.item()

        # backpropagation, optimizing and print and 
        loss.backward()
        optim.step()
        train_loss += loss.item()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {:4d} [{:10d}/{:10d} ({:3.0f}%)]\tLoss: {:12.6f}'.format(
                epoch, batch_idx * len(noisy_t), len(loader_train.dataset),
                100. * batch_idx / len(loader_train),
                loss.item() / len(noisy_t)))
            
    # save loss        
    Losses.train.append(train_loss / len(loader_train.dataset)) 
    if network_parameters.model == 'VAE':
        Losses.kld.append(kld_loss / len(loader_train.dataset))
        Losses.recon.append(recon_loss / len(loader_train.dataset))
        Losses.mse_train = (mse_train/len(loader_train.dataset))

    print('====> Epoch: {} Average loss Training: {:12.6f}'.format(epoch, Losses.train[-1]))

    return Losses

def test(network_parameters: NP, 
         loader_test: data.DataLoader, 
         optim: torch.optim.Optimizer, 
         model: torch.nn.Module,
         device: torch.device, 
         loss_function: torch.nn.modules.loss, 
         epoch: int, 
         h5pyFile: h5py.File, 
         Losses: LC) -> LC:
    """
    test function for neural networks, VAE or DAE.
    Also saves model checkpoints for usage and at final epoch count optimizer step additionally.

    Args:
        network_parameters (NP): an instance of the NetworkParameters util_classes, 
        loader_test (data.DataLoader): neural network data loader for training data, 
        optim (torch.optim.Optimizer): optimizer, 
        model (torch.nn.Module): model to be trained, 
        device (torch.device): device for training (cpu or cuda), 
        loss_function (torch.nn.modules.loss): loss function for reconstruction loss, 
        epoch (int): current training epoch, 
        Losses (LC): an instance of the Loss_class util_classes

    Returns:
        LC: Losses with added current test loss.

    """
    
    # initialize values and run as testing
    model.eval()
    kld_loss = 0.0
    recon_loss = 0.0
    mse_loss = 0.0
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (noisy_t, clean_t, D_clean, noise) in enumerate(loader_test):

            noisy_t = noisy_t.type(torch.FloatTensor).to(device)
            clean_t = clean_t.type(torch.FloatTensor).to(device)

            # denoise inputs and calculate loss (KLD irrelevant and irritating therefore not correctly handled)
            if network_parameters.model == 'DAE':
                recon_t = model(noisy_t)
                loss = loss_function(recon_t, clean_t)
            elif network_parameters.model == 'VAE':
                recon_t, mu, logvar = model(noisy_t)


                loss_lossf = loss_function(recon_t, clean_t)
                KLD = ae.loss_function_kld(mu, logvar)

                recon_loss += loss_lossf.item()
                loss = loss_lossf + KLD


            loss_mse = ae.loss_function_mse(recon_t, clean_t)
            mse_loss += loss_mse.item()
            test_loss += loss.item()

    # append losses to LC
    Losses.test.append(test_loss / len(loader_test.dataset))
    Losses.mse.append(mse_loss / len(loader_test.dataset))

    if network_parameters.model == 'VAE':
        Losses.testKld.append(kld_loss / len(loader_test.dataset))
        Losses.testRecon.append(recon_loss / len(loader_test.dataset))
        mu = mu.cpu().detach().numpy()
        Losses.means.append(mu)
        std = torch.exp(0.5 * logvar)
        std = std.cpu().detach().numpy()
        Losses.standards.append(std)

    print('====> Epoch: {} Average loss Testing: {:12.7f}'.format(epoch, Losses.test[-1]))
    print('====> Epoch: {} Average mse_loss Testing: {:12.7f}'.format(epoch, Losses.mse[-1]))

    #save current model state
    out_str=''
    if epoch == network_parameters.epochs:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + 'final.pt'
        out_str2 = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + 'final_trainable.tar'
        torch.save({'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':  optim.state_dict(),
                    'Losses': Losses,
                    'NetworkParameters': network_parameters},
                    network_parameters.acquisition_dir + out_str2)
    else:
        out_str = '/train_' + network_parameters.model + '_Latent' + str(network_parameters.latent).zfill(2) + '_epoch' + str(epoch).zfill(3) + '.pt'
    torch.save(model.state_dict(), network_parameters.acquisition_dir + out_str)
    
    return Losses

def setup(ACQ_DIR: str) -> tuple[NP, np.array, np.array, np.array, np.array]:
    """
    Setup function for the training script, loads configuration file for training and initializes all return variables:

    Args:
        ACQ_DIR (str): path to the directory containing yaml config and where to store data, 

    Returns:
        NP: network_parameters instance, storing all sorts of important parameters for training and set according to yaml config,
        np.array: simulated clean diffusion data,
        np.array: simulated clean diffusion data coefficents depending on model (BAS: stick coordinates, DTI: 3 eigenvectors of tensor),
        np.array: b0_mask, array where b0 entries of sequence b-values 0, all others 1, to later adapt the network architecture accordingly
        np.array: scaling_matrix, shape of bvals and certain factor for everyone, used to scale loss terms
    """

    stream = open(ACQ_DIR + '/config.yaml', 'r')
    config = yaml.load(stream, Loader)

    NetworkParameters = NP(config, ACQ_DIR)

    print(NetworkParameters)

    # load sequence info here, which stores bvals and gradient vectors
    # if different than usual data is used, please adapt here
    f = h5py.File(NetworkParameters.dvs_file_path, 'r')

    bvals = f['bvals'][:]
    bvecs = f['bvecs'][:]
    scalingMatrix = np.ones(bvals.shape)
    # scalingMatrix = scalingMatrix * bvals/1000
    f.close()

    #b0 values don't have to be learned, therefore mask is created
    b0_threshold = 50
    b0_mask = None
    if NetworkParameters.mask_usage == True:
        b0_mask = bvals > b0_threshold

    # simulate data using models 
    if NetworkParameters.diff_model == 'DTI':
        x_clean, original_D = dwi.model_DTI(bvals, bvecs, b0_threshold, 10, N_samples_first_evec=NetworkParameters.sphere_samples, N_samples_second_evec=int(NetworkParameters.sphere_samples*2/3))  #bvals = b, bvecs = g, returns y_pick
        original_D = original_D.T
        x_clean = x_clean.T
    elif NetworkParameters.diff_model == 'BAS':
        x_clean, original_D = dwi.model_BAS(bvals, bvecs, b0_threshold, N_samples=NetworkParameters.sphere_samples)  #bvals = b, bvecs = g, returns y_pick
        original_D = original_D.T
        x_clean = x_clean.T

    bvals = bvals[:, np.newaxis]

    return NetworkParameters, x_clean, original_D, b0_mask, scalingMatrix

def create_noised_dataset(train_set: dataset.qSpaceDataset, 
                          test_set: dataset.qSpaceDataset, 
                          NetworkParameters: NP) -> tuple[dataset.qSpaceDataset, dataset.qSpaceDataset]:
    """
    creates noisy versions of the handed over clean datasets. 
    Split of train and test set is done before to make the test samples completely unknown to the network in testing:

    Args:
        train_set (dataset.qSpaceDataset): train set for the network, 
        test_set (dataset.qSpaceDataset): test set for the network, 
        NetworkParameters (NP): network parameters storing important network information 

    Returns:
        NP: network_parameters instance, storing all sorts of important parameters for training and set according to yaml config,
        np.array: simulated clean diffusion data,
        np.array: simulated clean diffusion data coefficents depending on model (BAS: stick coordinates, DTI: 3 eigenvectors of tensor),
        np.array: B matrix from b values and g vectors,
        np.array: b0_mask, array where b0 entries of sequence b-values 0, all others 1, to later adapt the network architecture accordingly
        np.array: scaling_matrix, shape of bvals and certain factor for everyone, used to scale loss terms.
    """

    train_set_full = dc(train_set)
    test_set_full = dc(test_set)
    
    for id in range(1, NetworkParameters.noise_range, 1):
        train_set_copy = dc(train_set)
        test_set_copy = dc(test_set)

        # standard deviation of noise to be added, can be adjusted
        sd = 0.01 + id * 0.03
        print('noise sd = ', sd)

        # add noise to the copies
        for index in range(len(train_set_copy)):
            train_set_copy[index][0][:] = dataset.add_noise(train_set_copy[index][1], sd, NetworkParameters.noise_type)
            train_set_copy[index][3][:] = id

        for index in range(len(test_set_copy)):
            test_set_copy[index][0][:] = dataset.add_noise(test_set_copy[index][1], sd, NetworkParameters.noise_type)
            test_set_copy[index][3][:] = id

        # append noised versions to complete dataset
        train_set_full = data.ConcatDataset([train_set_full, train_set_copy])  
        test_set_full =  data.ConcatDataset([test_set_full, test_set_copy]) 

    print('>>train set size without noise', len(train_set))
    print('>>test set size without noise', len(test_set))
    print('>>train set size with noise', len(train_set)*NetworkParameters.noise_range)
    print('>>test set size with noise', len(test_set)*NetworkParameters.noise_range)

    return train_set_full, test_set_full

def create_data_loader(x_clean: np.array, 
                       original_D: np.array, 
                       NetworkParameters: NP) -> tuple[data.DataLoader, data.DataLoader]:
    """
    creates noisy versions of the handed over clean datasets. 
    Split of train and test set is done before to make the test samples completely unknown to the network in testing:

    Args:
        x_clean (dataset.qSpaceDataset): clean simulated diffusion data, 
        original_D (dataset.qSpaceDataset): simulated clean diffusion data coefficents depending on model (BAS: stick coordinates, DTI: 3 eigenvectors of tensor),
        NetworkParameters (NP): network parameters storing important network information 

    Returns:
        data.DataLoader: torch dataloader for training data to train network,
        data.DataLoader: torch dataloader for testing data to test network,
    """

    noise_amount = np.zeros(shape=(1, x_clean.shape[1]))
    noise_amount_full = np.transpose(noise_amount)

    x_clean = np.transpose(x_clean)
    x_no_noise_yet = dc(x_clean)
    original_D = np.transpose(original_D)

    print('>> shape x_clean = ', x_clean.shape)
    print('>> shape x_no_noise_yet = ', x_no_noise_yet.shape)
    print('>> shape original_D = ', original_D.shape)
    print('>> shape noise_amount_full = ', noise_amount_full.shape)

    # creates qSpaceDataset from simulated data, inherited from torch dataset and adding functionality
    q_dataset = dataset.qSpaceDataset(x_clean, x_no_noise_yet, original_D, noise_amount_full)
    q_datalen = len(q_dataset)

    # 80/20 % split of dataset for train/test set
    train_siz = int(q_datalen * 0.8)
    test_siz = q_datalen - train_siz
    train_set, test_set = data.random_split(q_dataset, [train_siz, test_siz])  

    # add noise to datasets
    train_set_noised, test_set_noised = create_noised_dataset(train_set, test_set, NetworkParameters)    

    loader_train = data.DataLoader(train_set_noised, batch_size=NetworkParameters.batch_size_train, shuffle=True)
    # test set not to be shuffeld to be able to compare results afterwards
    loader_test = data.DataLoader(test_set_noised, batch_size=NetworkParameters.batch_size_test, shuffle=False)

    return loader_train, loader_test

def main():
    """
    main script to train and test network.
    """

    torch.manual_seed(0)

    # reading in path to directory, where to store data and find config yaml
    given_dir = sys.argv[1] if len(sys.argv) > 1 else print('>> No directory handed over!')
    print(f'>> directory: {given_dir}\n')
    ACQ_DIR = given_dir

    # Setup training and testing data as well as network and training parameters and file to store losses
    NetworkParameters, x_clean, original_D, b0_mask, scalingMatrix = setup(ACQ_DIR)
    N_diff = scalingMatrix.shape[0]
    device = NetworkParameters.selectDevice()
    model = NetworkParameters.createModel(b0_mask, N_diff, device)
    loader_train, loader_test = create_data_loader(x_clean, original_D, NetworkParameters)
    loss_function = NetworkParameters.selectLoss()
    optimizer = NetworkParameters.selectOptimizer(model)
    Losses = LC(NetworkParameters.model)

    # file to store testing signals and denoised outputs over epochs
    f = h5py.File(ACQ_DIR + '/valid_' + NetworkParameters.model + '_Latent' + str(NetworkParameters.latent).zfill(2) + 'EpochTrain' + str(NetworkParameters.epochs) + '.h5', 'w')

    for epoch in range(1, NetworkParameters.epochs+1, 1):
        # training of netowrk with adaptive KLD update (see train method for more info)
        gamma_x = np.sqrt(Losses.mse_train)
        Losses = train(NetworkParameters, loader_train, optimizer, model, device, loss_function, epoch, Losses, gamma_x, scalingMatrix)
        

        if epoch == NetworkParameters.epochs or epoch % NetworkParameters.test_epoch_step == 0:
            # testing
            Losses = test(NetworkParameters, loader_test, optimizer, model, device, loss_function, epoch, f, Losses)

    # create file, where all kind of losses during training and testing are stored
    Losses.create_loss_file(NetworkParameters)
    # create mse loss file, shows the mse loss after testing for all checkpoints and the network architecture
    Losses.create_mse_loss_txt_file(NetworkParameters, model)

    f.close()

if __name__ == "__main__":
    main()