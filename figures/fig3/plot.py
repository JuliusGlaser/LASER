import h5py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from yaml import Loader
from latrec.utility.util import create_directory
from latrec.reconstruction.reconstruction import denoising_using_ae
import latrec.training.models.nn.autoencoder as ae
from latrec.training.sim import dwi
from latrec.training import linsub

#define variables of specific dataset
N_x = 200
N_y = 200
N_z = 114
N_q = 126
diff_model = 'BAS'
b0_threshold = 50
device = 'cpu'
BAS_dict = r'../../code/laser/training/trained_data/BAS/'
DTI_dict = r'../../code/laser/training/trained_data/DTI/'

stream = open(BAS_dict + 'config.yaml', 'r')
config = yaml.load(stream, Loader)

N_latent = config['latent']
N_layers = config['depth']
activ_fct = config['activation_fct']

q = 118 #direction to plot
z = 75


#load muse reconstructed slice 0

f = h5py.File(r'../../data/MUSE/MuseRecon_combined_slices.h5','r')
muse_dwi = f['DWI'][:]
muse_dwi = np.squeeze(muse_dwi)
f.close()
try:
    assert muse_dwi.shape == (N_q, N_z, N_y, N_x)
except:
    assert muse_dwi.T.shape == (N_q, N_z, N_y, N_x)
    muse_dwi = muse_dwi.T
print('>> muse dwi shape: ',muse_dwi.shape)

img = np.flipud(abs(muse_dwi[q,z,33:170,30:172]))
vmin = 0
vmax = np.max(muse_dwi[q,z,33:170,30:172])*0.7
create_directory('subfig_A')
print('>> plotting muse')
plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.savefig('subfig_A/Muse.png', bbox_inches='tight')
plt.savefig('subfig_A/Muse.pdf', bbox_inches='tight')

data_dict = {'subfig_D': '../../code/laser/denoising/denoised_comparison_noise_range_0.h5',
             'subfig_E': '../../code/laser/denoising/denoised_comparison_noise_range_8.h5'}
keys = ['DTI_SVD', 'BAS_SVD', 'DTI_VAE', 'BAS_VAE']

for fig_key in data_dict:
    print('>> plotting ' + fig_key + ' figures')
    file = h5py.File(data_dict[fig_key], 'r')
    create_directory(fig_key)
    for method in keys:
        print('>> plotting ' + method)
        img = np.rot90(abs(file[method][33:170,30:172,z,q]),1)
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.savefig(fig_key + os.sep + method + '.png', bbox_inches='tight', dpi=500)
        plt.savefig(fig_key + os.sep + method + '.pdf', bbox_inches='tight', dpi=500)