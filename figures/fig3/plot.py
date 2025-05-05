import h5py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from yaml import Loader
from laser.utility.util import create_directory

#define variables of specific dataset
N_x = 150
N_y = 150
N_z = 42
N_q = 126
diff_model = 'BAS'
b0_threshold = 50
device = 'cpu'
# BAS_dict = r'../../code/laser/training/trained_data/BAS/'
# DTI_dict = r'../../code/laser/training/trained_data/DTI/'


qvals = [21,39,120] #direction to plot
z = 31
y_slice = slice(15,127)
x_slice = slice(25,120)


#load muse reconstructed slice 0

f = h5py.File(r'../../data/ref/PI_CombShots.h5','r')
muse_dwi = f['DWI'][:]
muse_dwi = np.squeeze(muse_dwi)
f.close()
try:
    assert muse_dwi.shape == (N_q, N_z, N_y, N_x)
except:
    assert muse_dwi.T.shape == (N_q, N_z, N_y, N_x)
    muse_dwi = muse_dwi.T
print('>> muse dwi shape: ',muse_dwi.shape)

q = qvals[0]
print('q-value: ', q)
plt.clf()
create_directory('diff_' + str(q))
ref_img = np.flipud(abs(muse_dwi[q,z,y_slice,x_slice]))
vmin = 0
vmax = np.max(ref_img)*1
print('>> vmax: ', vmax)
create_directory('diff_' + str(q) + '/subfig_A')
print('>> plotting ref')
plt.imshow(ref_img, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
# plt.savefig('diff_' + str(q) + '/subfig_A/Ref.png', bbox_inches='tight', dpi=500)
plt.savefig('diff_' + str(q) + '/subfig_A/Ref.pdf', bbox_inches='tight', dpi=500)

create_directory('diff_' + str(q) + '/subfig_B')
f = h5py.File('/home/woody/mfqb/mfqb102h/meas_MID00201_FID00639_ep2d_diff_1_seg_3x1_126/reco_us_factor2_slice_015.h5', 'r')
ref_us = f['DWI'][:]
f.close()
ref_us = np.flipud(abs(ref_us[q, 0, y_slice, x_slice]))
plt.imshow(ref_us, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
# plt.savefig('diff_' + str(q) + '/subfig_B/Ref_us.png', bbox_inches='tight', dpi=500)
plt.savefig('diff_' + str(q) + '/subfig_B/Ref_us.pdf', bbox_inches='tight', dpi=500)

data_dict = {'diff_' + str(q) + '/subfig_C': '../../code/laser/denoising/denoised_comparison_reco_us_f2_slice_015.h5'}
z = 0
keys = ['DTI_SVD', 'BAS_SVD', 'DTI_VAE', 'BAS_VAE']

for fig_key in data_dict:
    print('>> plotting ' + fig_key + ' figures')
    file = h5py.File(data_dict[fig_key], 'r')
    create_directory(fig_key)
    for method in keys:
        print('>> plotting ' + method)
        dwi_ = file[method][:].T
        img = np.flipud(abs(dwi_[q, z, y_slice, x_slice]))
        plt.clf()
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        # plt.savefig(fig_key + os.sep + method + '.png', bbox_inches='tight', dpi=500)
        plt.savefig(fig_key + os.sep + method + '.pdf', bbox_inches='tight', dpi=500)
        
    file.close()

data_dict = {'diff_' + str(q) + '/subfig_differences': '../../code/laser/denoising/denoised_comparison_reco_us_f2_slice_015.h5'}
keys = ['DTI_SVD', 'BAS_SVD', 'DTI_VAE', 'BAS_VAE']
scaling = 2
for fig_key in data_dict:
    print('>> plotting ' + fig_key + ' figures')
    file = h5py.File(data_dict[fig_key], 'r')
    create_directory(fig_key)
    for method in keys:
        print('>> plotting ' + method)
        dwi_ = file[method][:].T
        img = np.flipud(abs(dwi_[q, z, y_slice, x_slice]))
        plt.imshow((ref_us - img)*scaling, cmap='gray', vmin=-vmax, vmax=vmax)
        plt.axis('off')
        rmse = np.sqrt(np.mean((ref_img - img) ** 2))
        # plt.title('RMSE = ' + str(rmse))
        # plt.savefig(fig_key + os.sep + method + '.png', bbox_inches='tight', dpi=500)
        plt.savefig(fig_key + os.sep + method + '_us.pdf', bbox_inches='tight', dpi=500)
        
        print('RMSE = ', rmse)
    file.close()

plt.imshow((ref_img - ref_us)*scaling, cmap='gray', vmin=-vmax, vmax=vmax)
plt.axis('off')
rmse = np.sqrt(np.mean((ref_img - ref_us) ** 2))
print('>> us_ref - ref')
print('RMSE = ', rmse)
# plt.title('RMSE = ' + str(rmse))
# plt.savefig('diff_' + str(q) + '/subfig_differences/Ref_us_ref.png', bbox_inches='tight', dpi=500)
plt.savefig('diff_' + str(q) + '/subfig_differences/Ref_us_ref.pdf', bbox_inches='tight', dpi=500)