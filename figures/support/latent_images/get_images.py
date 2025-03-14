import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
from latrec.utility.util import create_directory


slice_idx = 61

N_x = 200
N_y = 200
N_z = 114
N_lat = 11
std_shape = (N_x,N_y,N_z,N_lat)
vmaxes = [] 
vmines = []

DTI_path = r'../../data/LASER/DTI/DecRecon_combined_slices.h5'
BAS_path = r'../../data/LASER/BAS/DecRecon_combined_slices.h5'

dict = {'DTI':{'path':DTI_path, 'plt_title':'DTI/Latent_img_'}, 'BAS':{'path':BAS_path, 'plt_title':'BAS/Latent_img_'}}

for key in dict:
    print(key)
    create_directory(key)
    f = h5py.File(dict[key]['path'], 'r')
    DWI_latent = f['DWI_latent'][:]
    f.close()
    DWI_latent = DWI_latent.squeeze()

    print('>> DWI_latent.shape = ', DWI_latent.shape)
    
    for i in range(N_lat):
        img = np.flipud(DWI_latent[i, slice_idx, 22:186, 30:172])
        plt.imshow(img, cmap='grey', vmin=-1, vmax=+1)
        plt.axis('off')
        plt.savefig(dict[key]['plt_title'] + str(i) + '.png', bbox_inches='tight', dpi=500)
        plt.savefig(dict[key]['plt_title'] + str(i) + '.pdf', bbox_inches='tight', dpi=500)

#safe colorbar
# Define vmin and vmax
vmin, vmax = -1, 1
cmap = 'gray'  # Using 'grey' or 'gray' colormap

# Create a dummy scalar mappable for the colorbar
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create the figure and colorbar
fig, ax = plt.subplots(figsize=(1, 6))  # Adjust size for vertical colorbar
cb = plt.colorbar(sm, cax=ax, orientation='vertical')  # Change to 'horizontal' if needed

# Save the colorbar as PNG and PDF
plt.savefig('colorbar.png', bbox_inches='tight', dpi=500)
plt.savefig('colorbar.pdf', bbox_inches='tight', dpi=500)


