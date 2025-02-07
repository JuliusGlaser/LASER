import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from yaml import Loader
import argparse
from sigpy.mri import retro, app, sms, muse, mussels

from latrec.reconstruction.reconstruction import denoising_using_ae
import latrec.training.models.nn.autoencoder as ae
from latrec.training.sim import dwi
from latrec.training import linsub


parser = argparse.ArgumentParser(description="Parser to overwrite slice_idx and slice_inc")
parser.add_argument("--slice_idx", type=int, default=-1, help="Slice_idx to reconstruct")
parser.add_argument("--slice_inc", type=int, default=1, help="slice increment if multiple slice recon")
args = parser.parse_args()
slice_idx = args.slice_idx
slice_inc = args.slice_inc


#define variables of specific dataset
N_x = 200
N_y = 200
N_z = MB = 3
N_q = 126
N_slices = 114
diff_model = 'BAS'
b0_threshold = 50
device = 'cpu'
BAS_dict = '/home/hpc/mfqb/mfqb102h/tech_note_vae_diffusion/latrec/training/trained_data/more_iso_signals/'

stream = open(BAS_dict + 'config.yaml', 'r')
config = yaml.load(stream, Loader)

N_latent = config['latent']
N_layers = config['depth']
activ_fct = config['activation_fct']

N_slices_collap = N_slices // N_z


#load bvals and bvecs

f = h5py.File('/home/hpc/mfqb/mfqb102h/tech_note_vae_diffusion/latrec/raw-data/data-126-dir/1.0mm_126-dir_R3x3_dvs.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()
b0_mask = bvals > b0_threshold




#load VAE model

model_BAS = ae.VAE(b0_mask=b0_mask, input_features=N_q, latent_features=N_latent, depth=N_layers, activ_fct_str=activ_fct).to(device)
model_BAS.load_state_dict(torch.load(BAS_dict + 'train_VAE_Latent' +str(N_latent).zfill(2) + 'final.pt', map_location=torch.device(device),weights_only=True))
        
model_BAS = model_BAS.float()

for param in model_BAS.parameters():
    param.requires_grad = False


recons_all_slices_dwi = np.zeros((N_y, N_x, N_slices, N_q), dtype=np.complex_)
latents_all_slices_dwi = np.zeros((N_y, N_x, N_slices, N_latent), dtype=np.complex_)

if slice_idx >= 0:
    slice_loop = range(slice_idx, slice_idx + slice_inc, 1)
else:
    slice_loop = range(N_slices_collap)

for s in slice_loop:
    slice_str = str(s).zfill(3)
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

    #load muse reconstructed slice 0

    f = h5py.File('/home/woody/mfqb/mfqb102h/muse/MuseRecon_combined_slices.h5','r')
    muse_dwi = f['DWI'][:].T
    muse_dwi = np.squeeze(muse_dwi)
    f.close()

    #denoise muse data using VAE

    BAS_denoised, BAS_latent = denoising_using_ae(muse_dwi, (N_q, N_slices, N_x, N_y), model_BAS, N_latent, b0_mask, 'VAE', device)
    BAS_denoised = BAS_denoised.T
    BAS_latent = BAS_latent.T

    recons_all_slices_dwi[:,:,:,:] = BAS_denoised
    latents_all_slices_dwi[:,:,:,:] = BAS_latent

    
f = h5py.File('denoised_muse_VAE_BAS.h5','w')
f.create_dataset('DWI', data=np.array(recons_all_slices_dwi))
f.create_dataset('BAS_latent', data=np.array(latents_all_slices_dwi))
f.close()


