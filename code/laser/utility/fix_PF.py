import h5py
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp


mask = np.ones((1,1,110,110), dtype=bool)
mask[:, :, 0:27, :] = 0

f = h5py.File('/home/vault/mfqb/mfqb102h/LASER_bipolar_revision/kdat_us3/LASER/DAE_BAS_splitted_reco_new_scalings/DAE_BAS_splitted_reco_new_scalings_comb_part1.h5', 'r')
DWI1 = f['DWI'][:]
DWI_latent1 = f['DWI_latent'][:]
f.close()

f = h5py.File('/home/vault/mfqb/mfqb102h/LASER_bipolar_revision/kdat_us3/LASER/DAE_BAS_splitted_reco_new_scalings/DAE_BAS_splitted_reco_new_scalings_comb_part2.h5', 'r')
DWI2 = f['DWI'][:]
DWI_latent2 = f['DWI_latent'][:]
f.close()

DWI1 = sp.fft(DWI1, axes=(-2,-1))
DWI2 = sp.fft(DWI2, axes=(-2,-1))

DWI1 = DWI1*mask
DWI2 = DWI2*mask

DWI1 = sp.ifft(DWI1, axes=(-2,-1))
DWI2 = sp.ifft(DWI2, axes=(-2,-1))

f = h5py.File('/home/vault/mfqb/mfqb102h/LASER_bipolar_revision/kdat_us3/LASER/DAE_BAS_splitted_reco_new_scalings/DAE_BAS_splitted_reco_new_scalings_PF_comb_part1.h5', 'w')
f.create_dataset('DWI', data=DWI1)
f.create_dataset('DWI_latent', data=DWI_latent1)
f.close()

f = h5py.File('/home/vault/mfqb/mfqb102h/LASER_bipolar_revision/kdat_us3/LASER/DAE_BAS_splitted_reco_new_scalings/DAE_BAS_splitted_reco_new_scalings_PF_comb_part2.h5', 'w')
f.create_dataset('DWI', data=DWI2)
f.create_dataset('DWI_latent', data=DWI_latent2)
f.close()