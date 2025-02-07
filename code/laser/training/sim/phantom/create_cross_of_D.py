import numpy as np
import h5py
from deepsubspacemri.sim import dwi

def from_9_to_6(D):
    D_compressed = [D[0][0], D[0][1], D[1][1], D[0][2], D[1][2], D[2][2]] #Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    return D_compressed

#rd = right down
#lu = left up

D_lu_rd = np.array([[1,1,0],[1,1,0],[0,0,0.5]])
D_ru_ld = np.array([[1,-1,0],[-1,1,0],[0,0,0.5]])

maxRow = 100
width = 20
f = h5py.File('D_file.h5','w')

cross_lu_rd = np.zeros((maxRow,maxRow,3,3))
cross_ru_ld = np.zeros((maxRow,maxRow,3,3))




for row in range(maxRow):
    cross_lu_rd[row:row+width//2, row:row+width//2,:,:] =  D_lu_rd # Vertical line

f.create_dataset('D_half_cross_left', data=cross_lu_rd)

for row in range(maxRow):
    cross_ru_ld[row:row+width//2, maxRow-row-width//2:maxRow-row,:,:] = D_ru_ld  # Vertical line

f.create_dataset('D_half_cross_right', data=cross_ru_ld)

# combined

combined = cross_lu_rd + cross_ru_ld

f.create_dataset('D_full_cross', data=combined)

# create diffusion signal for 21 dir
# get g and b
g_and_b_file = h5py.File('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/data_files/4shot_data/1.0mm_21-dir_R1x3_dvs.h5', 'r')

#f = h5py.File(ACQ_DIR + '/3shell_126dir_diff_encoding.h5', 'r')

bvals = g_and_b_file['bvals'][:]
bvals = bvals[:, np.newaxis]
bvecs = g_and_b_file['bvecs'][:]

g_and_b_file.close()

# DTI

DTI_grid = np.zeros((100,100,21))

for x in range(DTI_grid.shape[0]):
    for y in range(DTI_grid.shape[1]):
        D1 = from_9_to_6(cross_lu_rd[x,y,:])
        D2 = from_9_to_6(cross_ru_ld[x,y,:])
        if np.any(D1) > 0 and np.any(D2) > 0:
            DTI_grid[x,y,:] = dwi.calc_DTI_res(bvals, bvecs, D1, 2E-3)*0.5 + dwi.calc_DTI_res(bvals, bvecs, D2, 2E-3)*0.5
        elif np.any(D1) > 0 and not np.any(D2) > 0:
            DTI_grid[x,y,:] = dwi.calc_DTI_res(bvals, bvecs, D1, 2E-3)
        else:
            DTI_grid[x,y,:] = dwi.calc_DTI_res(bvals, bvecs, D2, 2E-3)

f.create_dataset('dwi_DTI', data=DTI_grid)

BAS_grid = np.zeros((100,100,21))

for x in range(BAS_grid.shape[0]):
    for y in range(BAS_grid.shape[1]):
        D1 = from_9_to_6(cross_lu_rd[x,y,:])
        D2 = from_9_to_6(cross_ru_ld[x,y,:])
        BAS_grid[x,y,:] = dwi.calc_BAS_res(bvals, bvecs, D1, D2, 2E-3)

f.create_dataset('dwi_BAS', data=BAS_grid)
f.close()





