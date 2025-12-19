import h5py
import os
import numpy as np
import sigpy as sp
from sigpy.mri import sms
from os.path import exists
from pathlib import Path
import numpy as np
import yaml
from yaml import Loader
import argparse

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to config.yaml in the same folder as the script
config_path = os.path.join(script_dir, "config.yaml")

stream = open(config_path, 'r')
config = yaml.load(stream, Loader)

reco_data_path = config['reco_data_path']
name = config['name']
name_to_save = config['name_to_save']
dvs_file_path = config['dvs_file_path']

N_latent = config['N_latent']

N_diff = config['N_diff']
N_coils = config['N_coils']
N_y = config['N_y']
N_x = config['N_x']
MB = config['MB']
Accel_PE = config['Accel_PE']
N_slices = config['N_slices']
N_segments = config['N_segments']    

run_combine_DWI = config['run_combine_DWI']
run_combine_latent = config['run_combine_latent']
run_fit = config['run_fit']

parser = argparse.ArgumentParser(description='Read in slice files, append them, and save in correct order.')

parser.add_argument('--dir', default=reco_data_path,
                    help='directory in which the data are read.')

parser.add_argument('--method', default=name,
                    help='recon method.')

parser.add_argument('--out_name', default=name_to_save,
                    help='recon method.')

parser.add_argument('--MB', type=int, default=MB,
                    help='multi-band factor.')

parser.add_argument('--slices', type=int, default=N_slices,
                    help='total number of slices.')

parser.add_argument('--part', type=int, default=1,
                    help='Part.')
args = parser.parse_args()

N_slices = args.slices
MB = args.MB
reco_data_path = args.dir
name = args.method
name_to_save = args.out_name
part = args.part

# raw slice to get sequence parameters

maxInd = int(N_slices/MB)

if run_combine_DWI:
    slice_loop = range(0, maxInd, 1)
    recons_all_slices_dwi = np.zeros((N_diff, N_slices, N_y, N_x), dtype=np.complex64)
    for s in slice_loop:
        slice_str = str(s).zfill(3)
        f = h5py.File(reco_data_path +name + '_slice_' + slice_str + '_'+str(part)+'.h5', 'r')

        dwi_data = f['DWI'][:].squeeze()
        f.close()
        slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)
        for i in range(MB):
            n_slice = slice_mb_idx[i]
            if MB >1:
                recons_all_slices_dwi[:,n_slice,:,:] = dwi_data[:,i,:,:]
            else:
                recons_all_slices_dwi[:,n_slice,:,:] = dwi_data[:,:,:]
    #TODO: check if file is available already
    f = h5py.File(reco_data_path +  name_to_save + '.h5', 'w')
    f.create_dataset(name='DWI', data=recons_all_slices_dwi)
    f.close()

if run_combine_latent:
    recons_all_slices_dwi_latent = np.zeros((N_latent, N_slices, N_y, N_x), dtype=np.float64)
    slice_loop = range(0, maxInd, 1)
    for s in slice_loop:
        slice_str = str(s).zfill(3)
        f = h5py.File(reco_data_path +name + '_slice_' + slice_str + '_'+str(part)+'.h5', 'r')
        dwi_data = f['DWI_latent'][:].squeeze()
        f.close()
        # print(dwi_data.shape)
        slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)
        for i in range(MB):
            n_slice = slice_mb_idx[i]
            if MB > 1:
                recons_all_slices_dwi_latent[:,n_slice,:,:] = dwi_data[:,i,:,:]
            else:
                recons_all_slices_dwi_latent[:,n_slice,:,:] = dwi_data[:,:,:]

    #TODO: check if file is available already
    f = h5py.File(reco_data_path +  name_to_save + '.h5', 'r+')
    f.create_dataset(name='DWI_latent', data=recons_all_slices_dwi_latent)
    f.close()

if run_fit:
    f = h5py.File(reco_data_path+  name_to_save + '.h5', 'r+')
    f2 = h5py.File(dvs_file_path, 'r')
    bvals = f2['bvals'][:]
    bvecs = f2['bvecs'][:]
    f2.close()

    expected_shape = (N_x, N_y, N_slices, N_diff)
    # bvals = bvals.reshape(-1, 1)
    # B = epi.get_B(bvals, bvecs)

    print(bvals.shape)
    print(bvecs.shape)

    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals=bvals, bvecs=bvecs, atol=0.1)

    import dipy.reconst.dti as dti
    from dipy.reconst.dti import fractional_anisotropy, color_fa
    tenmodel = dti.TensorModel(gtab)


    dwi = f['DWI'][:]

    dwi = abs(np.squeeze(dwi)) * 1000   #scaling results in better fits
    print(dwi.shape)
    try:
        assert dwi.shape == expected_shape
    except:
        if dwi.shape == expected_shape[::-1]:
            dwi = dwi.T
        assert dwi.shape == expected_shape
    N_y, N_x, N_z, N_diff = dwi.shape


    b0 = np.mean(abs(dwi), axis=-1)
    id = b0 > np.amax(b0) * 0.01

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

    # f = h5py.File(r'C:\Workspace\LASER\data\LLR\JETS2.h5', 'r+')
    f.create_dataset('fa', data=FA)
    f.create_dataset('cfa', data=RGB)
    # f.create_dataset('DWI', data=dwi/1000)
    f.close()