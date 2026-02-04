"""
This module implements a denoising comparison between AE trained with diffusion tensor and ball-and-stick datasets
and the linear subspace approach of learning a singular values threshold by looking at a simulated signal dictionary
and mapping data to this created subspace of singular values

Authors:
    Julius Glaser <julius-glaser@gmx.de>
"""

import os
import h5py
import numpy as np
import torch
import yaml
from yaml import Loader
from copy import deepcopy as dc
from pathlib import Path
import importlib.util

from laser.reconstruction.reconstruction import denoising_using_ae
import laser.training.models.nn.autoencoder as ae
from laser.training.sim import dwi
from laser.training import linsub

def add_noise(x_clean, scale, noiseType = 'gaussian'):

    if noiseType== 'gaussian':
        x_noisy = x_clean + np.random.normal(loc = 0,
                                            scale = scale,
                                            size=x_clean.shape)
    elif noiseType== 'rician':
        noise1 =np.random.normal(0, scale, size=x_clean.shape)
        noise2 = np.random.normal(0, scale, size=x_clean.shape)
        x_noisy =  np.sqrt((x_clean + noise1) ** 2 + noise2 ** 2)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy

stream = open('config.yaml', 'r')
config = yaml.load(stream, Loader)

#define variables of dataset to denoise
N_x = config['N_x']                                                     # X samples
N_y = config['N_y']                                                     # Y samples
N_z = config['N_z']                                                     # slices of data (of file or complete combined data)
N_q = config['N_q']                                                     # diffusion encodings of data
b0_threshold = config['b0_threshold']                                   # b0 threshold
device = config['device']                                               # device to run denoising on
BAS_dir = config['BAS_dir']                                             # path to directory of VAE BAS model checkpoint
DTI_dir = config['DTI_dir']                                             # path to directory of VAE DTI model checkpoint
dvs_file_path = config['dvs_file_path']                                 # path to dvs file
muse_data_path = config['muse_data_path']                               # path to MUSE data
Dictionary_sphere_samples = config['Dictionary_sphere_samples']         # Samples from unit sphere for dictionary data (meaning depends on diffusion model)

NR = config['NR']                                                       # Noise distributions to be added to simulated dictionary (0 or 1 leads to no noise)

if device == 'cuda':
    #check if cuda is available
    if not torch.cuda.is_available():
        print('CUDA is not available, using CPU instead')
        device = 'cpu'

#load muse reconstructed data (can be combined or just one slice)

p = Path(muse_data_path)
f = h5py.File(p,'r')
muse_dwi = f['DWI'][:]
# muse_dwi = np.squeeze(muse_dwi)
f.close()

print(muse_dwi.shape)
try:
    assert muse_dwi.shape == (N_q, N_z, N_y, N_x)
except:
    assert muse_dwi.T.shape == (N_q, N_z, N_y, N_x)
    muse_dwi = muse_dwi.T
print('>> muse dwi shape: ',muse_dwi.shape)

path = p.parent / "denoised_results"
path_str = str(path) + os.sep 
os.makedirs(path_str, exist_ok=True)

file_path = os.path.join(path_str, 'Data_denoised.h5')
file_to_save = h5py.File(file_path, 'w')

print('>> Saving denoised results to: ', file_path)

#load bvals and bvecs

f = h5py.File(dvs_file_path, 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()
print(bvals.dtype)
b0_mask = bvals > b0_threshold


#load VAE model BAS
stream = open(BAS_dir + 'config.yaml', 'r')
config = yaml.load(stream, Loader)
N_latent = config['latent']
N_layers = config['depth']
activ_fct = config['activation_fct']
modelType = config['model']

ae_dict = {'DAE':ae.DAE, 
        'VAE':ae.VAE}

model_BAS = ae_dict[modelType](b0_mask=b0_mask, input_features=N_q, latent_features=N_latent, depth=N_layers, activ_fct_str=activ_fct).to(device)
model_BAS.load_state_dict(torch.load(BAS_dir + 'train_'+modelType+'_Latent' +str(N_latent).zfill(2) + 'final.pt', map_location=torch.device(device), weights_only=True))
        
model_BAS = model_BAS.float()

for param in model_BAS.parameters():
    param.requires_grad = False

#denoise muse data using VAE
print(bvals.dtype)
BAS_denoised, BAS_latent = denoising_using_ae(muse_dwi, (N_q, N_z, N_y, N_x), model_BAS, N_latent, modelType, device, bvals=bvals)
BAS_denoised = BAS_denoised.T
BAS_latent = BAS_latent.T

print('>> BAS denoised shape: ', BAS_denoised.shape)
file_to_save.create_dataset('BAS_AE', data=BAS_denoised)
file_to_save.create_dataset('BAS_AE_lat', data=BAS_latent)

#generate dictionary data using Ball-and-stick model

BAS_x_clean, BAS_original_D = dwi.model_BAS(bvals, bvecs, b0_threshold, N_samples=Dictionary_sphere_samples)
BAS_original_D = BAS_original_D.T
BAS_x_clean = BAS_x_clean.T

BAS_full = BAS_x_clean

for id in range(1, NR, 1):
    # DTI_copy = dc(DTI_x_clean)
    BAS_copy = dc(BAS_x_clean)
    # standard deviation of noise to be added, can be adjusted
    sd = 0.01 + id * 0.03
    print('noise = ', sd)

    # add noise to the copies
    BAS_copy = add_noise(BAS_copy, sd)

    # append noised versions to complete dataset
    BAS_full = np.append(BAS_full, BAS_copy, axis=1)

# print('>> number of signals for DTI training linsub: ',DTI_full.shape[1])
print('>> number of signals for BAS training linsub: ',BAS_full.shape[1])


#load DAE model DTI
stream = open(DTI_dir + 'config.yaml', 'r')
config = yaml.load(stream, Loader)
N_latent = config['latent']
N_layers = config['depth']
activ_fct = config['activation_fct']
modelType = config['model']

model_DTI = ae_dict[modelType](b0_mask=b0_mask, input_features=N_q, latent_features=N_latent, depth=N_layers, activ_fct_str=activ_fct).to(device)
model_DTI.load_state_dict(torch.load(DTI_dir + 'train_'+modelType+'_Latent' +str(N_latent).zfill(2) + 'final.pt', map_location=torch.device(device), weights_only=True))

model_DTI = model_DTI.float()

for param in model_DTI.parameters():
    param.requires_grad = False

#denoise muse data using DAE

DTI_denoised, DTI_latent = denoising_using_ae(muse_dwi, (N_q, N_z, N_y, N_x), model_DTI, N_latent, modelType, device, bvals=bvals)
DTI_denoised = DTI_denoised.T
DTI_latent = DTI_latent.T

print('>> DTI denoised shape: ', DTI_denoised.shape)
print('>> latent shape: ', BAS_latent.shape)

file_to_save.create_dataset('DTI_AE', data=DTI_denoised)
file_to_save.create_dataset('DTI_AE_lat', data=DTI_latent)

# #denoise muse data using subspace
dwi_scale = np.divide(muse_dwi, muse_dwi[0, ...],
                        out=np.zeros_like(muse_dwi),
                        where=muse_dwi!=0)

muse_dwi_torch = torch.tensor(dwi_scale, device=device, dtype=torch.float32)
print(muse_dwi_torch.shape)


# #train subspace model BAS
print('>> Run subspace denoising with error bound equal to no noise DAE_BAS training (0.00055)')

BAS_full_tensor = torch.tensor(BAS_full).to(device).to(torch.float)
print('>> BAS_full_tensor shape: ',BAS_full_tensor.shape)
BAS_linsub_basis_tensor_00055_eb = linsub.learn_linear_subspace(BAS_full_tensor, num_coeffs=N_latent, error_bound = 0.00055, use_error_bound=True, device=device)
print('>> BAS_linsub_basis_tensor shape: ',BAS_linsub_basis_tensor_00055_eb.shape)
print(BAS_linsub_basis_tensor_00055_eb.dtype)

BAS_dwi_linsub_tensor_00055_eb = BAS_linsub_basis_tensor_00055_eb @ BAS_linsub_basis_tensor_00055_eb.T @ abs(muse_dwi_torch).contiguous().view(N_q, -1)

BAS_dwi_linsub_tensor_00055_eb = BAS_dwi_linsub_tensor_00055_eb.view(muse_dwi_torch.shape)

BAS_dwi_linsub_00055_eb = BAS_dwi_linsub_tensor_00055_eb.detach().cpu().numpy()
BAS_linsub_denoised_00055_eb = BAS_dwi_linsub_00055_eb * muse_dwi[0]
BAS_linsub_denoised_00055_eb = BAS_linsub_denoised_00055_eb.T

print('>> BAS_linsub_denoised_00055_eb denoised shape: ', BAS_linsub_denoised_00055_eb.shape)
file_to_save.create_dataset('BAS_SVD_00055_eb', data=BAS_linsub_denoised_00055_eb)


# print('>> Run subspace denoising with error bound of 0.00001')

# BAS_full_tensor = torch.tensor(BAS_full).to(device).to(torch.float)
# print('>> BAS_full_tensor shape: ',BAS_full_tensor.shape)
# BAS_linsub_basis_tensor_00001_eb = linsub.learn_linear_subspace(BAS_full_tensor, num_coeffs=N_latent, error_bound = 0.00001, use_error_bound=True, device=device)
# print('>> BAS_linsub_basis_tensor shape: ',BAS_linsub_basis_tensor_00001_eb.shape)
# print(BAS_linsub_basis_tensor_00001_eb.dtype)

# BAS_dwi_linsub_tensor_00001_eb = BAS_linsub_basis_tensor_00001_eb @ BAS_linsub_basis_tensor_00001_eb.T @ abs(muse_dwi_torch).contiguous().view(N_q, -1)

# BAS_dwi_linsub_tensor_00001_eb = BAS_dwi_linsub_tensor_00001_eb.view(muse_dwi_torch.shape)

# BAS_dwi_linsub_00001_eb = BAS_dwi_linsub_tensor_00001_eb.detach().cpu().numpy()
# BAS_linsub_denoised_00001_eb = BAS_dwi_linsub_00001_eb * muse_dwi[0]
# BAS_linsub_denoised_00001_eb = BAS_linsub_denoised_00001_eb.T

# print('>> BAS_linsub denoised shape: ', BAS_linsub_denoised_00001_eb.shape)
# file_to_save.create_dataset('BAS_SVD_00001_eb', data=BAS_linsub_denoised_00001_eb)


print('>> Run subspace denoising with fixed 11 singular values')

BAS_full_tensor = torch.tensor(BAS_full).to(device).to(torch.float)
print('>> BAS_full_tensor shape: ',BAS_full_tensor.shape)
BAS_linsub_basis_tensor_ss_11 = linsub.learn_linear_subspace(BAS_full_tensor, num_coeffs=N_latent, use_error_bound=False, device=device)
print('>> BAS_linsub_basis_tensor shape: ',BAS_linsub_basis_tensor_ss_11.shape)
print(BAS_linsub_basis_tensor_ss_11.dtype)

BAS_dwi_linsub_tensor_ss_11 = BAS_linsub_basis_tensor_ss_11 @ BAS_linsub_basis_tensor_ss_11.T @ abs(muse_dwi_torch).contiguous().view(N_q, -1)

BAS_dwi_linsub_tensor_ss_11 = BAS_dwi_linsub_tensor_ss_11.view(muse_dwi_torch.shape)

BAS_dwi_linsub_ss_11 = BAS_dwi_linsub_tensor_ss_11.detach().cpu().numpy()
BAS_linsub_denoised_ss_11 = BAS_dwi_linsub_ss_11 * muse_dwi[0]
BAS_linsub_denoised_ss_11 = BAS_linsub_denoised_ss_11.T

print('>> BAS_linsub denoised shape: ', BAS_linsub_denoised_ss_11.shape)
file_to_save.create_dataset('BAS_SVD_ss_11', data=BAS_linsub_denoised_ss_11)


file_to_save.close()