import argparse
import h5py
import nibabel as nib
import numpy as np
import os

DIR = os.path.dirname(os.path.realpath(__file__))
print(DIR)

# %%
parser = argparse.ArgumentParser(description='convert .h5 to .nii')

parser.add_argument('--input_file',
                    default='MUSE_cplx_denoise',
                    help='input .nii file name')

args = parser.parse_args()

# %%
DAT_DIR = DIR

img = nib.load(args.input_file + '.nii')
DWI = np.asanyarray(img.dataobj)
print('DWI shape: ', DWI.shape)

f = h5py.File(args.input_file + '.h5', 'w')
f.create_dataset('DWI', data=DWI)
f.close()
