# plots comparison between reconstructed b0* and b0**

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt


slice_idx = 1  # sms slice 11

BAS_path = r'../../data/LASER/BAS/DecRecon_slice_030.h5'

f = h5py.File(BAS_path, 'r')
b0 = f['b0'][:]
b0 = b0[0,slice_idx,:,:]
b0_comb = f['b0_combined'][:]
b0_comb = b0_comb[slice_idx,:,:]
f.close()

img = np.flipud(np.real(b0[30:172,22:186]))
vmax = np.max(img)*0.7
vmin = np.min(img)
plt.imshow(img, cmap='gray', vmax=vmax, vmin=vmin)
plt.axis('off')
plt.savefig('b0.png', bbox_inches='tight', dpi=500)
plt.savefig('b0.pdf', bbox_inches='tight', dpi=500)

img = np.flipud(np.real(b0_comb[30:172,22:186]))
plt.imshow(img, cmap='gray', vmax=vmax, vmin=vmin)
plt.axis('off')
plt.savefig('b0_comb.png', bbox_inches='tight', dpi=500)
plt.savefig('b0_comb.pdf', bbox_inches='tight', dpi=500)




