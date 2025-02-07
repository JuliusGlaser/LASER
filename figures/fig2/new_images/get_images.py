import h5py
import matplotlib.pyplot as plt
import numpy as np
from sigpy.mri import retro, app, sms, muse, mussels
import latrec.training.models.nn.autoencoder as ae
from latrec.utility.util import *
import yaml
from yaml import Loader
import torch

load_fonts()
set_figure_style(label_size=30)

path_to_reco = r'W:\radiologie\mrt-probanden\AG_Laun\Julius Glaser\LASER_data\dec\\'
path_to_raw = r'W:\radiologie\mrt-probanden\AG_Laun\Julius Glaser\LASER_data\raw\\'
reco_slice = 30
reco_file = 'DecRecon_slice_' + str(reco_slice).zfill(3) + '.h5'
raw_file = '1.0mm_126-dir_R3x3_kdat_slice_' + str(reco_slice).zfill(3) + '.h5'
slices = [1]

N_x, N_y, N_z, N_diff = (200,200,3,126)
N_latent = 11
N_coils = 32

#Number of images to save
images_diff = 4
images_latent = 4
images_kspace = 4

f = h5py.File(path_to_reco+reco_file, 'r')
dwi = f['DWI'][:].squeeze()
latent = f['DWI_latent'][:]
b0 = f['b0_combined'][:]
f.close()
f = h5py.File(path_to_raw+raw_file,'r')
kspace = f['kdat'][:].squeeze()
MB = f['MB'][()]
N_slices = f['Slices'][()]
N_segments = f['Segments'][()]
N_Accel_PE = f['Accel_PE'][()]
f.close()
kspace = np.swapaxes(kspace, -2, -3)
# kdat_prep = []
# for d in range(N_diff):
#     k = retro.split_shots(kspace[d, ...], shots=N_segments)
#     kdat_prep.append(k)

# kdat_prep = np.array(kdat_prep)

assert dwi.shape == (N_diff,N_z,N_x,N_y)

assert latent.shape == (N_z*N_x*N_y,N_latent)
latent = latent.reshape((N_z,N_x,N_y,N_latent))

# assert kdat_prep.shape == (N_diff, N_segments, N_coils, N_y, N_x)

assert b0.shape == (N_z,N_x,N_y)

# for slic in slices:
#     for diff in range(images_diff):
#         plt.imshow(np.rot90(abs(dwi[1+diff,slic,:,:]),2),vmax=0.0001, vmin=0,cmap='gray')
#         plt.axis('off')
#         plt.savefig('DWI_dir_'+str(1+diff)+'_slice_'+str(reco_slice)+'_z_'+str(slic)+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
#     for lat in range(images_latent):
#         plt.imshow(np.rot90(abs(latent[slic,:,:,lat]),2),vmax=1, vmin=0,cmap='gray')
#         plt.axis('off')
#         plt.savefig('latent_dir_'+str(lat)+'_slice_'+str(reco_slice)+'_z_'+str(slic)+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
#     for s in range(N_segments):
#         for k in range(images_kspace):
#             for c in range(4):
#                 plt.imshow(abs(kdat_prep[k,s,c,:,:]),vmax=0.00004, vmin=0,cmap='gray')
#                 plt.axis('off')
#                 plt.savefig('ksp_dir_'+str(k)+'_slice_'+str(reco_slice)+'_coil_'+str(c)+'_shot_'+str(s)+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
#     plt.imshow(np.rot90(abs(b0[slic,:,:]),2),vmax=0.0004, vmin=0,cmap='gray')
#     plt.axis('off')
#     plt.savefig('b0_slice_'+str(reco_slice)+'_z_'+str(slic)+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
        
# save signals
x_ind = 125
y_ind = 113
z_ind = 1

# Sizes of your plots
size1 = 11
size2 = 22

# Base width for the smaller plot
base_width = 3
base_hight = 1.5
markersize= 2

# Calculate the scaling factor for the second plot
scaling_factor = 0.4
plt.figure(figsize=(base_width* scaling_factor, base_hight))

plt.plot(latent[z_ind,x_ind,y_ind,:], color='#e9ab47', marker='o', markersize=markersize, zorder=3)
plt.yticks([], [])
plt.xticks([], [])
# Draw x and y axes (zero lines)
plt.axhline(0, color='black', linewidth=1.0, zorder=1)  # Horizontal axis at y=0
plt.axvline(-1, color='black', linewidth=1.0, zorder=1)  # Vertical axis at x=0

# Remove all other spines
for spine in ['top', 'right', 'left', 'bottom']:
    plt.gca().spines[spine].set_visible(False)
# plt.xlabel("Latent neurons ($\\beta$)")
# plt.ylabel("Signal")
# plt.axis('off')
plt.savefig('latent_signal_'+str(reco_slice)+'_z'+'.pdf', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.savefig('latent_signal_'+str(reco_slice)+'_z'+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.clf()
#unscaled decoded signal

diff_model = 'BAS'
b0_threshold = 50
device = 'cpu'
BAS_dict = r'C:\Workspace\tech_note_vae_diffusion\latrec\training\trained_data\more_iso_signals\\'

stream = open(BAS_dict + 'config.yaml', 'r')
config = yaml.load(stream, Loader)

N_latent = config['latent']
N_layers = config['depth']
activ_fct = config['activation_fct']

N_slices_collap = N_slices // N_z


#load bvals and bvecs

f = h5py.File(r'C:\Workspace\tech_note_vae_diffusion\latrec\raw-data\data-126-dir\1.0mm_126-dir_R3x3_dvs.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()
b0_mask = bvals > b0_threshold

model_BAS = ae.VAE(b0_mask=b0_mask, input_features=N_diff, latent_features=N_latent, depth=N_layers, activ_fct_str=activ_fct).to(device)
model_BAS.load_state_dict(torch.load(BAS_dict + 'train_VAE_Latent' +str(N_latent).zfill(2) + 'final.pt', map_location=torch.device(device)))
        
model_BAS = model_BAS.float()

for param in model_BAS.parameters():
    param.requires_grad = False

x_1 = torch.tensor(latent[z_ind,:,:,:], dtype=torch.float).to(device)
print(x_1.shape)
x_1 = x_1.reshape((200*200,11))
out = model_BAS.decode(x_1)
out = out.reshape((200,200,N_diff))
out = out.numpy()
    
scaling_factor2 = 2
plt.figure(figsize=(base_width*scaling_factor2, base_hight))
plt.plot(out[x_ind,y_ind,:], color='red', marker='o', markersize=markersize, zorder=3)
plt.yticks([], [])
plt.xticks([], [])
# Draw x and y axes (zero lines)
plt.axhline(0, color='black', linewidth=1.0, zorder=1)  # Horizontal axis at y=0
plt.axvline(-5, color='black', linewidth=1.0, zorder=1)  # Vertical axis at x=0

# Remove all other spines
for spine in ['top', 'right', 'left', 'bottom']:
    plt.gca().spines[spine].set_visible(False)
# plt.xlabel("Diffusion encodings (q)")
# plt.ylabel("Signal")
# plt.axis('off')
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # Customize as needed
plt.savefig('unscaled_decoded_signal_'+str(reco_slice)+'_z'+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.savefig('unscaled_decoded_signal_'+str(reco_slice)+'_z'+'.pdf', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.clf()

for diff in range(images_diff):
    plt.imshow(np.rot90(out[:,:,1+diff],2),cmap='gray')
    plt.axis('off')
    plt.savefig('decoded_dir_'+str(1+diff)+'_slice_'+str(reco_slice)+'_z.png', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.clf()
#scaled signal

plt.figure(figsize=(base_width*scaling_factor2, base_hight))
plt.plot(abs(dwi[:,z_ind,x_ind,y_ind]), color='red', marker='o', markersize=markersize)
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel("Diffusion encodings (q)")
plt.ylabel("Signal")
# plt.axis('off')
plt.savefig('scale_signal_'+str(reco_slice)+'_z'+'.png', bbox_inches='tight', dpi=300, pad_inches = 0)
plt.clf()