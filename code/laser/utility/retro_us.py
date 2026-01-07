"""
This script implements retrospective undersampling to fully sampled (PF) data.

Author: Julius Glaser <julius-glaser@gmx.de>
"""


import argparse
import h5py
import os
import pathlib
import torch
import twixtools
import scipy.io

import numpy as np
import sigpy as sp

# from sigpy import extract_dicom
from sigpy.mri import app, epi, sms, util

def create_directory(path: str)->bool:
    """
    Creates a directory at the specified path if it doesn't already exist.

    Parameters:
    path (str): The directory path to create.

    Returns:
    bool: True if the directory was created, False if it already exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")
        return True
    else:
        print(f"Directory already exists at: {path}")
        return False

parser = argparse.ArgumentParser(description='run reconstruction.')

parser.add_argument('--dir',
                    default='Dir/to/data',
                    help='directory in which the data are read.')

parser.add_argument('--data_name', default='kdat_slice',
                    help='Kspace data name.')

parser.add_argument('--us', type=int, default=2,
                    help='Undersampling factor')

parser.add_argument('--split_diff', action='store_true',
                    help='split diffusion (assumes 2 averages).')

args = parser.parse_args()


us_factor = args.us
DIR = args.dir

f = h5py.File(DIR + os.sep + args.data_name + '_000.h5', 'r')
# ['MB', 'Segments', 'Slices', 'iPat', 'kdat', 'slice_idx']
MB = f['MB'][()]
N_slices = f['Slices'][()]
Segments = f['Segments'][()]
iPat = f['iPat'][()]
slice_idx = f['slice_idx'][()]
f.close()

out_dir = DIR + os.sep + 'kdat_us' + str(us_factor)
create_directory(out_dir)

for sli in range(N_slices):

    print('> slice ' + str(sli).zfill(3))

    fstr = DIR + os.sep + args.data_name + '_' + str(sli).zfill(3)

    f = h5py.File(fstr + '.h5', 'r')
    kdat = f['kdat'][:]
    f.close()

    print('kdat shape: ', kdat.shape)

    # us k-space
    for i in range(us_factor-1):
        kdat[..., i::us_factor, :, :] = 0

    # save us k-space
    outprefstr = out_dir + os.sep + args.data_name + '_' + str(sli).zfill(3) +'_us' + str(us_factor) + '.h5'

    f = h5py.File(outprefstr, 'w')
    if args.split_diff is True:
        if kdat.shape[7] % 2 != 0:
            raise ValueError('Cannot split odd number of diffusion directions.')
        n_diff = int(kdat.shape[7] / 2)
        kdat1 = kdat[:,:,:,:,:,:,:, :n_diff, ...]
        kdat2 = kdat[:,:,:,:,:,:,:, n_diff:, ...]
        f.create_dataset('kdat1', data=kdat1)
        f.create_dataset('kdat2', data=kdat2)
    else:
        f.create_dataset('kdat', data=kdat)
    f.create_dataset('MB', data=MB)
    f.create_dataset('Slices', data=N_slices)
    f.create_dataset('Segments', data=Segments)
    f.create_dataset('iPat', data=iPat)
    f.create_dataset('slice_idx', data=slice_idx)
    f.close()