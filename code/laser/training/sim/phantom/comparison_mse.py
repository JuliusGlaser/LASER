"""
This module implements the VAE training for DW-MRI

Authors:

    Julius Glaser <julius.glaser@fau.de>
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
import deepsubspacemri.linsub as linsub
import copy

# from torch.utils.data import DataLoader
from deepsubspacemri.sim import dataset
from deepsubspacemri.sim import dwi
from deepsubspacemri.models.nn import autoencoder as ae
from deepsubspacemri.sim import epi

from deepsubspacemri.train import Network_parameters
from deepsubspacemri.train import Losses_class
from deepsubspacemri.train import test
from deepsubspacemri.train import setup
from deepsubspacemri.train import create_noised_dataset
from deepsubspacemri.train import create_data_loader


def main():

    DIR = os.path.dirname(os.path.realpath(__file__))

    DATA_DIR = '/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/data/'

    torch.manual_seed(0)
    
    given_dir = sys.argv[1] if len(sys.argv) > 1 else print('No directory handed over!')

    print(f'> directory: {given_dir}\n')

    ACQ_DIR = DATA_DIR + '/' + given_dir

    NetworkParameters, x_clean, original_D, B, b0_mask = setup(ACQ_DIR)
    test_set = h5py.File('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/data_files/train_and_test_set_21_new_split_real.h5', 'r')
    clean = test_set['test_set_clean'][:]
    noisy = test_set['test_set_noisy'][:]
    D_clean = test_set['test_set_D'][:]
    noise = test_set['test_set_noise'][:]

    q_dataset = dataset.qSpaceDataset(noisy[:, 1:], clean[:,1:], D_clean, noise)

    loader_test = data.DataLoader(q_dataset, batch_size=1, shuffle=False)

    device = NetworkParameters.selectDevice()

    BAS_model = ae.VAE(b0_mask, input_features=NetworkParameters.N_diff, latent_features=NetworkParameters.latent).to(NetworkParameters.device)
    BAS_model.load_state_dict(torch.load('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/sim/phantom/BAS_with_ball_6Latent.pt', map_location=torch.device(NetworkParameters.device)))

    DTI_model = ae.VAE(b0_mask, input_features=NetworkParameters.N_diff, latent_features=NetworkParameters.latent).to(NetworkParameters.device)
    DTI_model.load_state_dict(torch.load('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/sim/phantom/DTI_6Latent.pt', map_location=torch.device(NetworkParameters.device)))

    # model = NetworkParameters.createModel(q_dataset.N_diff, device)
    loss_function = NetworkParameters.selectLoss()

    DTI_file = h5py.File(ACQ_DIR + '/valid_' + NetworkParameters.model + '_Latent' + str(NetworkParameters.latent).zfill(2) + 'EpochTrain' + str(NetworkParameters.epochs) + 'DTI.h5', 'w')
    BAS_file = h5py.File(ACQ_DIR + '/valid_' + NetworkParameters.model + '_Latent' + str(NetworkParameters.latent).zfill(2) + 'EpochTrain' + str(NetworkParameters.epochs) + 'BAS.h5', 'w')
    

    #Create txt config file:
    completeName = os.path.join(ACQ_DIR, "mseLoss.txt")

    txtFile = open(completeName, "w")

    txtFile.write("\n\n\n Results in Testing:\n\n")

    Losses_DTI = Losses_class(NetworkParameters.model)
    Losses_BAS = Losses_class(NetworkParameters.model)

    Losses_DTI = test(NetworkParameters, loader_test, DTI_model, device, loss_function, 1, DTI_file, Losses_DTI)
    Losses_BAS = test(NetworkParameters, loader_test, BAS_model, device, loss_function, 1, BAS_file, Losses_BAS)

    linsub_basis_tensor = linsub.learn_linear_subspace(torch.tensor(clean.T[1:,:]), num_coeffs=NetworkParameters.latent, use_error_bound=False, device=device)

    test_linsub = []
    mse_loss = 0.0

    linsubFile = h5py.File(ACQ_DIR + '/linsub_test_same_latent.h5', 'w')

    for batch_idx, (noisy_t, clean_t, D_clean, noise) in enumerate(loader_test):
        # tensor_to_add = torch.tensor([[1]])
        # noisy_t = torch.cat((tensor_to_add, noisy_t), dim=1)
        # clean_t = torch.cat((tensor_to_add, clean_t), dim=1)
        linsub_tensor = linsub_basis_tensor @ linsub_basis_tensor.T @ noisy_t.view(NetworkParameters.N_diff-1, -1)

        linsub_recon = linsub_tensor.view(noisy_t.shape)
        test_linsub.append(linsub_recon.detach().cpu().numpy())
        loss_mse = ae.loss_function_mse(linsub_recon, clean_t)
        mse_loss += loss_mse

    linsubFile.create_dataset('linsubRecon' + str(NetworkParameters.epochs), data=np.array(test_linsub))
    print(f'------- MSE Loss for linsub: {mse_loss/len(loader_test)} --------')
    linsubFile.close()
    
    NetworkParameters.model = 'VAE_DTI'
    Losses_DTI.create_loss_file(NetworkParameters)   
    NetworkParameters.model = 'VAE_BAS'
    Losses_BAS.create_loss_file(NetworkParameters) 
    NetworkParameters.model = 'VAE'
            
    txtFile.write('\n\n')
    txtFile.write(BAS_model.__str__())
    txtFile.close()
    
    DTI_file.close()    
    BAS_file.close()

if __name__ == "__main__":
    main()