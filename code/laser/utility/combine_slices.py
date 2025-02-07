import argparse
import h5py
import os
import nibabel as nib

import numpy as np
import sigpy as sp

from sigpy.mri import retro, app, sms, muse, mussels
from os.path import exists

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
path = '/home/woody/mfqb/mfqb102h/DTI_special_layers/' 
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
f.close()
quit()
#convert to nifti

f = h5py.File(path +  name + '_combined_slices.h5', 'r')
DWI = f['DWI'][:].T
f.close()

assert DWI.shape == (200,200,114,126)

DWI = np.squeeze(DWI*10**6)


print('DWI shape: ', DWI.shape)

DWI_abs = np.abs(DWI)
DWI_abs.astype(np.int16)
DWI_phs = np.angle(DWI)

img_abs = nib.Nifti1Image(DWI_abs, affine=np.eye(DWI.ndim))
nib.save(img_abs, path +name + '_combined' + '_abs.nii.gz')

img_phs = nib.Nifti1Image(DWI_phs, affine=np.eye(DWI.ndim))
nib.save(img_phs, path + name + '_combined' + '_phs.nii')
