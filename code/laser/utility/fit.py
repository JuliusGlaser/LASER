import os
import h5py

import nibabel as nib
import numpy as np
from pathlib import Path


# %% b-values and vectors
f = h5py.File('/home/hpc/mfqb/mfqb102h/tech_note_vae_diffusion/latrec/raw-data/data-126-dir/1.0mm_126-dir_R3x3_dvs.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()

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


# %% 2. loop over all source files to fit MD, FA, and RGB
path = r'/home/woody/mfqb/mfqb102h/denoised/denoised_b0_combined.h5'
f = h5py.File(path, 'r+')
dwi = f['DWI'][:]

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