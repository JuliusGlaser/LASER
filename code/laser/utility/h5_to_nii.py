import argparse
import h5py
import nibabel as nib
import numpy as np
import os

# DIR = os.path.dirname(os.path.realpath(__file__))
# print(DIR)

# %%
parser = argparse.ArgumentParser(description='convert .h5 to .nii')

parser.add_argument('--input_file',
                    default='MUSE',
                    help='input .h5 file name')

parser.add_argument('--input_key',
                    default='DWI',
                    help='input .h5 file key')

parser.add_argument('--output_file',
                    default='',
                    help='output .nii file name')

args = parser.parse_args()

if args.output_file == '':
    args.output_file = args.input_file
# %%

f = h5py.File(args.input_file + '.h5', 'r')
DWI = f[args.input_key][:].T
if DWI.ndim > 4:
    print('DWI has more than 4 dimensions, squeezing to 4D')
    DWI = np.squeeze(DWI)
if DWI.ndim != 4:
    print('DWI has less than 4 dimensions, expanding to 4D')
    DWI = DWI[..., np.newaxis,:]
f.close()

# DWI = np.squeeze(DWI)

print('DWI shape: ', DWI.shape)

DWI_abs = np.abs(DWI)*1000
DWI_phs = np.angle(DWI)

# affine=np.array([[-1.81818, 0, 0, 100.612],
#                    [0, 1.81818, 0, -93.01261],
#                    [0, 0, 3.75, -20.7228],
#                    [0, 0, 0, 1]])
affine=np.eye(DWI.ndim)

img_abs = nib.Nifti1Image(DWI_abs, affine=affine)
img_abs.header.set_sform(affine, code=1)
img_abs.header.set_qform(affine, code=1)
nib.save(img_abs, args.output_file + '_abs.nii')


img_phs = nib.Nifti1Image(DWI_phs, affine=affine)
nib.save(img_phs, args.input_file + '_phs.nii')

# %%
