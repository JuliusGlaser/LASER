import argparse
import h5py
import os
import nibabel as nib

import numpy as np
import sigpy as sp

from sigpy.mri import retro, app, sms, muse, mussels
from os.path import exists
from pathlib import Path

import numpy as np

import yaml
from yaml import Loader


f  = h5py.File('/home/woody/mfqb/mfqb102h/raw/1.0mm_3-shell_R3x3_kdat_slice_032.h5', 'r')  
kdat = f['kdat'][:]
MB = f['MB'][()]
accel_factor = f['Accel_PE'][()]
N_slices = f['Slices'][()]
N_segments = f['Segments'][()]    
f.close()

kdat = np.squeeze(kdat)  # 4 dim
kdat = np.swapaxes(kdat, -2, -3)
N_diff, N_coils, N_x, N_y = kdat.shape

maxInd = 37
slice_loop = range(0, maxInd+1, 1)
path = '/home/woody/mfqb/mfqb102h/no_special_layers/b0_combined/BAS/' 
name = 'DecRecon'
lam = 0

recons_all_slices_dwi = np.zeros((N_diff, N_slices, N_x,N_y), dtype=np.complex_)
slice_factor = N_slices//MB-1
if lam is None:
    lamda = 0
else:
    lamda = lam
for s in slice_loop:
    slice_str = str(s).zfill(3)
    f = h5py.File(path +name + '_slice_' + slice_str + '.h5', 'r')




    dwi_data = f['DWI'][:].squeeze()
    f.close()
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)
    for i in range(MB):
        n_slice = slice_mb_idx[i]




        recons_all_slices_dwi[:,n_slice,:,:] = dwi_data[:,i,:,:]
f = h5py.File(path +  name + '_combined_slices.h5', 'w')


f.create_dataset(name='DWI', data=recons_all_slices_dwi)




# %% b-values and vectors
f2 = h5py.File('/home/hpc/mfqb/mfqb102h/tech_note_vae_diffusion/latrec/raw-data/data-126-dir/1.0mm_126-dir_R3x3_dvs.h5', 'r')
bvals = f2['bvals'][:]
bvecs = f2['bvecs'][:]
f2.close()

expected_shape = (200,200,114,126)
# bvals = bvals.reshape(-1, 1)
# B = epi.get_B(bvals, bvecs)

print(bvals.shape)
print(bvecs.shape)

from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals=bvals, bvecs=bvecs, atol=0.1)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
tenmodel = dti.TensorModel(gtab)


dwi = recons_all_slices_dwi

dwi = abs(np.squeeze(dwi)) * 1000
print(dwi.shape)
try:
    assert dwi.shape == expected_shape
except:
    if dwi.shape == expected_shape[::-1]:
        dwi = dwi.T
    assert dwi.shape == expected_shape
N_x, N_y, N_z, N_diff = dwi.shape


b0 = np.mean(abs(dwi), axis=-1)
id = b0 > np.amax(b0) * 0.01
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

FA  = (FA.T).T
RGB = (RGB.T).T
MD  = (MD.T).T

f.create_dataset('fa', data=FA)
f.create_dataset('cfa', data=RGB)
f.close()