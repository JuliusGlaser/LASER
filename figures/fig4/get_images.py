import numpy as np
import os
import h5py
import matplotlib.pyplot as plt


slice_idx = 29

q_1000 = 2
q_2000 = 40
q_3000 = 57
q_values = [q_1000, q_2000, q_3000]
N_x = 220
N_y = 223
N_z = 42
N_q = 126
std_shape = (N_x,N_y,N_z,N_q)
vmaxes = [] 
vmines = []
img_width = 142
img_height = 205
ratio = img_width/img_height
plt_title = 'comparison_muse_llr_denoised_DTI_BAS'


muse_path = r'../../data/MUSE/MUSE.h5'
llr_path = r'../../data/LLR/JETS2.h5'
denoised_path = r'../../data/denoised/VAE_BAS_denoised.h5'
DTI_path = r'../../data/LASER/VAE_DTI_joint/DecRecon_combined_slices.h5'
BAS_path = r'../../data/LASER/VAE_BAS_joint/DecRecon_combined_slices.h5'

dict = {'MUSE':{'path':muse_path, 'row':0}, 'LLR':{'path':llr_path, 'row':1}, 'denoised':{'path':denoised_path, 'row':2},
         'DTI':{'path':DTI_path, 'row':3}, 'BAS':{'path':BAS_path, 'row':4}}

plt.rcParams['figure.figsize'] = [len(dict)*ratio,len(q_values)+1]
fig, axes = plt.subplots(len(dict), len(q_values)+1, constrained_layout=False, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})


for key in dict:
    print(key)
    f = h5py.File(dict[key]['path'], 'r')
    dwi = f['DWI'][:]
    if key == 'DTI' or key=='BAS':
        dwi = dwi.T
    try:
        cfa = f['cfa'][:]
    except:
        cfa = f['cFA'][:]
    f.close()
    dwi = dwi.squeeze()
    try:
        assert dwi.shape == std_shape
    except:
        if dwi.T.shape == std_shape:
            dwi = dwi.T

    print('>>DWI.shape = ', dwi.shape)

    try:
        cfa.shape == (N_x,N_y,N_z,3)
    except:
        if cfa.T.shape == (N_x,N_y,N_z,3):
            cfa = cfa.T
    print('>>cfa.shape = ', cfa.shape)
    
    for q in range(len(q_values)):
        img = np.rot90(abs(dwi[35:175,22:186,slice_idx,q_values[q]]),1)
        if key == 'MUSE':
            #define vmax, vmin according to Muse image
            vmaxes.append(np.max(img))
            vmines.append(np.min(img))
        axes[dict[key]['row'],q].imshow(img, cmap='gray', vmax=vmaxes[q], vmin=vmines[q])
        axes[dict[key]['row'],q].axis('off')
    cfa_img = np.rot90(cfa[35:175,22:186,slice_idx,:])
    axes[dict[key]['row'],-1].imshow(cfa_img)
    axes[dict[key]['row'],-1].axis('off')


plt.savefig(plt_title + '.png', bbox_inches='tight', dpi=500)
plt.savefig(plt_title + '.pdf', bbox_inches='tight', dpi=500)

