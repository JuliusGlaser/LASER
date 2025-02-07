import numpy as np
import random
import h5py
from scipy.io import savemat
from scipy.ndimage import binary_dilation
from deepsubspacemri.sim.dwi import rotTensorAroundX as RTAX
from deepsubspacemri.sim.dwi import rotTensorAroundY as RTAY
from deepsubspacemri.sim.dwi import rotTensorAroundZ as RTAZ
from deepsubspacemri.sim import dwi

Tensor0 = np.array([0, 0, 0, 0, 0, 0])
TensorX = np.array([1, 0, 0.5, 0, 0, 0.5])
TensorY = np.array([0.5, 0, 1, 0, 0, 0.5])
TensorZ = np.array([0.5, 0, 0.5, 0, 0, 1])

tensor_1_field = np.zeros((40,40,6))
tensor_2_field = np.zeros((40,40,6))

rot_ang = np.arange(0,180,5)


tensor1 = TensorX
tensor2 = TensorX

file = h5py.File('TensorFieldsDict.h5','w')
dict = {}
for rot in rot_ang:
    tensor1 = tensor1
    tensor2 = RTAZ(tensor2, -rot)
    # fill upper 3/4 of tensor_1_field with tensor1 and 3/4 of tensor_2_filed with tensor2
    tensor_1_field[0:30,:,:] = tensor1
    tensor_2_field[11::,:,:] = tensor2
    grp = file.create_group(f'angle_{rot}')
    grp.create_dataset('tensor_1_field', data=tensor_1_field)
    grp.create_dataset('tensor_2_field', data=tensor_2_field)
    if rot == 0 or rot== 90 or rot== 180:
        mdic = {"matrix1": tensor_1_field,"matrix2": tensor_2_field, "label": "Fields"}
        savemat('fields_rot_' + str(rot) +'.mat', mdic)

file.close()

