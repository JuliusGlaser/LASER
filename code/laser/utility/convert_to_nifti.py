import h5py
import os
import nibabel as nib
import numpy as np
import sigpy as sp
from os.path import exists

import numpy as np

#convert to h5 file to nifti
data_path = r'C:\local_laser_data\potporri\LLR' + os.sep #path to data which shall be converted
name = 'JETS2'
expected_shape = (220,223,42,126) # N_x, N_y, N_slices, N_diffusion_encodings
data_key = 'DWI'


f = h5py.File(data_path + os.sep + name + '.h5', 'r')
DWI = f[data_key][:]
f.close()
print('DWI shape: ', DWI.shape)
N_x, N_y, N_z, N_q = expected_shape

try:
    assert DWI.shape == expected_shape
except:
    if DWI.shape[::-1] == expected_shape:
        DWI = DWI.T

DWI = np.squeeze(DWI*10**6)


print('DWI shape: ', DWI.shape)

DWI_abs = np.abs(DWI)
DWI_abs.astype(np.int16)
DWI_phs = np.angle(DWI)

img_abs = nib.Nifti1Image(DWI_abs, affine=np.eye(DWI.ndim))
nib.save(img_abs, data_path + os.sep + name + '_abs.nii.gz')

img_phs = nib.Nifti1Image(DWI_phs, affine=np.eye(DWI.ndim))
nib.save(img_phs, data_path + os.sep + name + '_phs.nii')