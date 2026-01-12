import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
f = h5py.File(r'../../data/raw/1.0mm_126-dir_R3x3_dvs.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()
gtab = gradient_table(bvals, bvecs, atol=0.1)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
tenmodel = dti.TensorModel(gtab)


def fit(dwi):
    dwi = abs(np.squeeze(dwi)).T * 1000
    print(dwi.shape)

    N_x, N_y, N_z, N_diff = dwi.shape

    dwi = dwi[:,:,:,:]
    b0 = np.mean(abs(dwi), axis=-1)
    id = b0 > np.amax(b0) * 0.01
    mask = np.zeros_like(b0)
    mask[id] = 1
    # b0_mask, mask = median_otsu(b0,
    #                             median_radius=4,
    #                             numpass=4)

    b1 = np.mean(abs(dwi[..., 1:]), axis=-1)


    tenfit = tenmodel.fit(dwi)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = np.squeeze(color_fa(FA, tenfit.evecs))
    MD = tenfit.md
    return RGB


slice_idx = 2

q_1000 = 2
q_2000 = 40
q_3000 = 57
q_values = [q_1000, q_2000, q_3000]
N_x = 220
N_y = 223
N_z = 3
N_q = 126
std_shape = (N_q,N_z,N_y, N_x)
plt_shape = tuple(reversed(std_shape))  #(N_x,N_y,N_z,N_q)
vmaxes = [] 
vmines = []
img_width = 142
img_height = 205
ratio = img_width/img_height
plt_title = 'comparison_muse_llr_denoised_DTI_BAS'


muse_path = r'../../data/Paper_request_data/HR/MUSE/MuseRecon_slice_000.h5'
llr_path = r'../../data/Paper_request_data/HR/LLR/LLR_slice_000.h5'
denoised_path = r'../../data/Paper_request_data/HR/MUSE/MuseRecon_slice_000_cplx_denoise.h5'
DTI_path = r'../../data/Paper_request_data/HR/LASER/DAE_DTI/DecRecon_slice_000_lam_0.01.h5'
BAS_path = r'../../data/Paper_request_data/HR/LASER/DAE_BAS/DecRecon_slice_000_lam_0.01.h5'

dict = {'MUSE':{'path':muse_path, 'row':0}, 'LLR':{'path':llr_path, 'row':1}, 'denoised':{'path':denoised_path, 'row':2},
         'DTI':{'path':DTI_path, 'row':3}, 'BAS':{'path':BAS_path, 'row':4}}

plt.rcParams['figure.figsize'] = [len(dict)*ratio,len(q_values)+1]
fig, axes = plt.subplots(len(dict), len(q_values)+1, constrained_layout=False, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})


for key in dict:
    print(key)
    f = h5py.File(dict[key]['path'], 'r')
    dwi = f['DWI'][:]
    f.close()

    if dwi.ndim > 4:
        print('DWI has more than 4 dimensions, squeezing to 4D')
        dwi = np.squeeze(dwi)
    if dwi.ndim != 4:
        print('DWI has less than 4 dimensions, expanding to 4D')
        dwi = dwi[..., np.newaxis,:]

    try:
        assert dwi.shape == std_shape
    except:
        if dwi.T.shape == std_shape:
            dwi = dwi.T
            
   
    cfa = fit(dwi)  

    print('>>DWI.shape = ', dwi.shape)

    try:
        cfa.shape == (N_x,N_y,N_z,3)
    except:
        if cfa.T.shape == (N_x,N_y,N_z,3):
            cfa = cfa.T
    print('>>cfa.shape = ', cfa.shape)

    try:
        assert dwi.shape == plt_shape
    except:
        if dwi.T.shape == plt_shape:
            dwi = dwi.T

    if key== 'denoised':
        dwi = dwi / 1000  # scale MPPCA data
    
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

