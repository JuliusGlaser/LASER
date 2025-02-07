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

#convert to nifti
path = r'W:\radiologie\mrt-probanden\AG_Laun\Julius Glaser\LASER_data\denoised_muse_VAE_BAS\\'
name = 'denoised_muse_VAE_BAS'
f = h5py.File(path + name + '.h5', 'r')
DWI = f['BAS_denoised'][:]
f.close()

print('DWI shape: ', DWI.shape)

expected_shape = (200,200,114,126)
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
nib.save(img_abs, path + name + '_abs.nii.gz')

img_phs = nib.Nifti1Image(DWI_phs, affine=np.eye(DWI.ndim))
nib.save(img_phs, path + name + '_phs.nii')