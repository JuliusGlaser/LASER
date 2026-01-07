import os
import h5py
import numpy as np
import gc
import dipy.reconst.sfm as sfm
from dipy.reconst.odf import gfa
import nibabel as nib

import dipy.data as dpd
from dipy.data import get_sphere

import dipy.direction.peaks as dpp

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    auto_response_msmt,
    mask_for_response_msmt,
    multi_shell_fiber_response,
    response_from_mask_msmt,
)
import dipy.reconst.shm as shm
from dipy.segment.mask import median_otsu
from dipy.direction import peaks_from_model
from dipy.segment.tissue import TissueClassifierHMRF
from PIL import Image
import matplotlib.pyplot as plt
import copy
from dipy.core.sphere import unit_icosahedron
import cvxpy
from itertools import permutations, product
# use high angular resolution sphere
sphere = unit_icosahedron.subdivide(n=6)#get_sphere(name="symmetric724")


import yaml
from yaml import Loader
from time import time
import warnings
import subprocess

class Parameters:
    def __init__(self, orientation: str, 
                 slice_ind: tuple, 
                 matlab_area_TL: tuple, 
                 matlab_area_DR: tuple, 
                 directory: str,
                 orientationDict,
                 dictionary):
        self.orientation = orientation
        self.slice_ind_low = slice_ind[0]-1
        self.slice_ind_high = slice_ind[1]-1
        self.matlab_area_TL = matlab_area_TL
        self.matlab_area_DR = matlab_area_DR
        # translate coordinates from Matlab (origin bottom right) to the origins in the odf Slicer and for the rectangle drawing in the end
        self.slicer_area_TL, self.slicer_area_DR = self.coosMatlab2Slicer(orientationDict['standard'])
        self.rect_area_TL, self.rect_area_DR = self.coosMatlab2Rect(orientationDict['standard'])
        self.directory = directory
        self.dictionary = dictionary
        
        self.ensureDirectoryExists()

    
    def coosMatlab2Slicer(self, standardShape):
        N_y,N_x,N_z,N_q = standardShape
        # TL = x0,y0 DR = x1,y1
        x0 = self.matlab_area_TL[1]-1
        y0 = self.matlab_area_TL[0]-1
        x1 = self.matlab_area_DR[1]-1
        y1 = self.matlab_area_DR[0]-1
        
        if self.orientation == 'tra':
            #TODO: check if actually working
            areaTLNew = (y0+1, N_x - x0)
            areaDRNew = (y1, N_x - x1-1)
        elif self.orientation == 'cor':
            areaTLNew = (N_z-y1, N_x -x1-1)
            areaDRNew = (N_z-y0-1, N_x -x0)
        return areaTLNew, areaDRNew
    
    def coosMatlab2Rect(self, standardShape):
        N_y,N_x,N_z,N_q = standardShape
        # TL = x0,y0 DR = x1,y1
        x0 = self.matlab_area_TL[1]-1
        y0 = self.matlab_area_TL[0]-1
        x1 = self.matlab_area_DR[1]-1
        y1 = self.matlab_area_DR[0]-1
        if self.orientation == 'tra':
            #Additional border shift due to rectangle definition
            areaTLNew = (y0+1, N_x - x0)
            areaDRNew = (y1-1, N_x - x1-2)
        elif self.orientation == 'cor':
            areaTLNew = (N_z-y0-1, N_x -x0)
            areaDRNew = (N_z-y1, N_x -x1 - 1)
        return areaTLNew, areaDRNew
    
    def ensureDirectoryExists(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def createConfigTxt(self):
        completeName = os.path.join(self.directory, "parameters.txt")

        txtFile = open(completeName, "w")

        txtFile.write("Parameters used:\n\n")
        txtFile.write(Parameters.__str__(self))
        txtFile.close()

    def __str__(self) -> str:
        output_str = ''
        output_str += f'> used orientation: {self.orientation}\n'
        output_str += f'> used slice_ind: {self.slice_ind}\n'
        output_str += f'> used matlab_area_TL: {self.matlab_area_TL}\n'
        output_str += f'> used matlab_area_DR: {self.matlab_area_DR}\n'
        return output_str


def remove_black_borders(image_path, output_path, template=None, template_path=None):
    # Open the image
    if template:
        tmp = Image.open(template_path).convert("RGBA")
        img = Image.open(image_path).convert("RGBA")
    else:
        img = Image.open(image_path).convert("RGBA")
        tmp = img
    
    # Get the dimensions of the image
    width, height = img.size

    # Define the boundaries for cropping
    left, top, right, bottom = 0, 0, width, height

    # Scan from the top to find the first non-black row
    for y in range(height):
        row = tmp.crop((0, y, width, y + 1))
        if not all(pixel == (0, 0, 0, 255) for pixel in row.getdata()):
            top = y
            break

    # Scan from the bottom to find the last non-black row
    for y in range(height - 1, -1, -1):
        row = tmp.crop((0, y, width, y + 1))
        if not all(pixel == (0, 0, 0, 255) for pixel in row.getdata()):
            bottom = y + 1
            break

    # Scan from the left to find the first non-black column
    for x in range(width):
        column = tmp.crop((x, 0, x + 1, height))
        if not all(pixel == (0, 0, 0, 255) for pixel in column.getdata()):
            left = x
            break

    # Scan from the right to find the last non-black column
    for x in range(width - 1, -1, -1):
        column = tmp.crop((x, 0, x + 1, height))
        if not all(pixel == (0, 0, 0, 255) for pixel in column.getdata()):
            right = x + 1
            break
    
    # Crop the image using the found boundaries
    cropped_img = img.crop((left, top, right, bottom))
    
    # Save the cropped image
    cropped_img.save(output_path)

def normalize_path(path):
    """
    Normalize a file path to be platform-independent and usable in Python.
    
    Parameters:
        path (str): The file path as copied from Windows or Linux.
        
    Returns:
        str: The normalized path.
    """
    # Replace backslashes with forward slashes for initial consistency
    path = path.replace("\\", "/")
    # Use os.path.normpath to make it platform-specific
    normalized_path = os.path.normpath(path)

    if not os.path.splitext(normalized_path)[1]:  # No file extension
        # Append a trailing slash or backslash based on the OS
        if not normalized_path.endswith(os.sep):
            normalized_path += os.sep

    return normalized_path

def load_data(key, path_to_data, dictionary, odf_calc, orientationDict):
    dataFile = h5py.File(path_to_data + key +  os.sep + dictionary[key]+'.h5', 'r')
    # if key == 'lpca':
    #     if odf_calc:
    #         data = abs(dataFile['DWI'][:])* 1000
    #     FA   = dataFile['FA_lam_0'][:]
    #     print('>> FA lpca shape:',FA.shape)
    data = abs(dataFile['DWI'][:].T)* 1000
    dataFile.close()

    # flip z-axis, because it is necessary from standard data shape #TODO: evaluate if necessary for tra and sag
    if odf_calc:
        
        try:
            print(data.shape)
            assert data.shape == tuple(orientationDict['standard'])
        except AssertionError as e:
            try:
                data = data.T
                assert data.shape == tuple(orientationDict['standard'])
                print('data was transposed')
                
            except AssertionError as e2:
                print("Assertion failed")
                raise

    return data

def minmax_normalize(samples, out=None):
    """Min-max normalization of a function evaluated on the unit sphere

    Normalizes samples to ``(samples - min(samples)) / (max(samples) -
    min(samples))`` for each unit sphere.

    Parameters
    ----------
    samples : ndarray (..., N)
        N samples on a unit sphere for each point, stored along the last axis
        of the array.
    out : ndrray (..., N), optional
        An array to store the normalized samples.

    Returns
    -------
    out : ndarray, (..., N)
        Normalized samples.

    """
    if out is None:
        dtype = np.common_type(np.empty(0, 'float32'), samples)
        out = np.array(samples, dtype=dtype, copy=True)
    else:
        out[:] = samples

    sample_mins = np.min(samples, -1)[..., None]
    sample_maxes = np.max(samples, -1)[..., None]
    out -= sample_mins
    out = np.divide(out, (sample_maxes - sample_mins),
                        out=np.zeros_like(out),
                        where=out!=0)
    return out

def load_data_bootstrap(key, path_to_data, dictionary, odf_calc, orientationDict, n='1'):
    full_data_path = path_to_data + key +  os.sep + dictionary[key]
    full_data_path = full_data_path[:-1]
    # check for existence of file
    # if not os.path.isfile(full_data_path+'.h5'):
    #     raise FileNotFoundError(f"File {full_data_path+n+'.h5'} does not exist.")
    dataFile = h5py.File(full_data_path+n+'.h5', 'r')
    # if key == 'lpca':
    #     if odf_calc:
    #         data = abs(dataFile['DWI'][:])* 1000
    #     FA   = dataFile['FA_lam_0'][:]
    #     print('>> FA lpca shape:',FA.shape)
    data = abs(dataFile['DWI'][:].T)* 1000
    dataFile.close()

    # flip z-axis, because it is necessary from standard data shape #TODO: evaluate if necessary for tra and sag
    if odf_calc:
        
        try:
            # print(data.shape)
            assert data.shape == tuple(orientationDict['standard'])
        except AssertionError as e:
            try:
                data = data.T
                assert data.shape == tuple(orientationDict['standard'])
                print('data was transposed')
                
            except AssertionError as e2:
                print("Assertion failed")
                raise

    return data

def angle_between_vectors(a, b):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta)

def calc_angles(GT_vector1, GT_vector2, PI_vector1, PI_vector2):
    ''' For entries where an entry is expected but none was fitte we return np.inf
        For entries where no entry is expected and none was fitted we return np.nan'''
    only1vecGT = False
    only1vecPI = False
    if np.array_equal(GT_vector2, np.array([0,0,0])) and np.array_equal(GT_vector1, np.array([0,0,0])):
        # print('Warning: no GT vectors!')
        return [np.nan, np.nan]
    if np.array_equal(PI_vector1, np.array([0,0,0])) and np.array_equal(PI_vector2, np.array([0,0,0])):
        # print('Warning: no reco vectors!')
        # if vector 1 doesn't exist there can't be a vector 2
        if np.array_equal(GT_vector2, np.array([0,0,0])) and not np.array_equal(GT_vector1, np.array([0,0,0])):
            return [np.nan, np.inf]
        else:
            return [np.inf, np.inf]
    
    if np.array_equal(GT_vector2, np.array([0,0,0])):
        GT_vectors = [GT_vector1]   # Search for only one vector in the PI-vectors
        only1vecGT = True
        # print('Warning: only one GT vector!')
    else:
        GT_vectors = [GT_vector1, GT_vector2]

    if only1vecGT:
        PI_vectors = [PI_vector1]
        only1vecPI = True           #FIXME: In theory there is a case where two are fitted but only one GT vector exists
        # print('Warning: only one PI vector!')
    elif np.array_equal(PI_vector2, np.array([0,0,0])):
        PI_vectors = [PI_vector1]
        # print('Warning: only one PI vector althoug 2 GT vectors!')
        only1vecPI = True
    else:
        PI_vectors = [PI_vector1, PI_vector2]

    # if only1vec:
    #     res_ang = [angle_between_vectors(GT_vector1, PI_vector1)]
    #     # if res_ang[0]>np.pi/2:
    #     #     res_ang[0] = np.pi - res_ang[0]
    # else:
    #     res_ang = [angle_between_vectors(GT_vector1, PI_vector1), angle_between_vectors(GT_vector2, PI_vector2)]
    #     # if res_ang[0]>np.pi/2:
    #     #     res_ang[0] = np.pi - res_ang[0]
    #     # if res_ang[1]>np.pi/2:
    #     #     res_ang[1] = np.pi - res_ang[1]
    best_combo = None
    smallest_ang = np.pi/2
    for pi_perm in permutations(PI_vectors):
        for signs in product([1, -1], repeat=2):
                angles = [angle_between_vectors(gt, s * pi) for gt, pi, s in zip(GT_vectors, pi_perm, signs)]
                if angles[0]>np.pi/2:
                    continue
                if not (only1vecGT or only1vecPI):
                    if angles[1]>np.pi/2:
                        continue
                if angles[0] < smallest_ang:
                    smallest_ang = angles[0]
                    best_combo = (pi_perm, signs, angles)             
                if not (only1vecGT or only1vecPI):
                    if angles[1] < smallest_ang:
                        smallest_ang = angles[1]
                        best_combo = (pi_perm, signs, angles)
    try:
        res_ang = best_combo[-1]
    except TypeError:
        print('No valid angle combination found!')
        print('GT vectors:', GT_vectors)
        print('PI vectors:', PI_vectors)
        res_ang = [np.inf, np.inf]
    if len(res_ang)==1 and only1vecGT:
        res_ang.append(np.nan)
    elif len(res_ang)==1 and only1vecPI:
        res_ang.append(np.inf)
    return np.array(res_ang)

def convert_to_nii(data, save_path):
    data = np.squeeze(data).T
    data_abs = np.abs(data)
    data_path = save_path + '_abs.nii'
    img_abs = nib.Nifti1Image(data_abs, affine=np.eye(data.ndim))
    nib.save(img_abs, data_path)
    return data_path

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

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    seq_path = normalize_path('..\\..\\..\\data\\raw\\1.0mm_126-dir_R3x3_dvs.h5')

    seqInfo = h5py.File(seq_path, 'r')
    bvals   = seqInfo['bvals'][:]
    bvecs   = seqInfo['bvecs'][:]
    seqInfo.close()

    bost_file = h5py.File('bootstrapping_vectors.h5', 'r')
    bost_vectors = bost_file['vectors'][:].squeeze()
    bost_file.close()

    N_bootstrap = config['N_bootstrap']
    SH_order = config['SH_order']
    min_separation_angle = config['min_separation_angle']
    rel_peak_threshold = config['rel_peak_threshold']


    print('Using N_bootstrap =', N_bootstrap)
    bost_vectors = bost_vectors[:N_bootstrap,...]

    gtab = gradient_table(bvals, bvecs, atol=3e-2)

    path_to_data = config['path_to_data']
    path_to_data = normalize_path(path_to_data)
    dictionary = config['dictionary']
    data_dict = {key: {'data':None,'FA':None} for key in dictionary}
    # dictionary = {'muse': 'MuseRecon_combined_slices', 'llr': 'JETS2', 'dec':'DecRecon_combined_slices','DTI': 'DecRecon_combined_slices'}

    orientationDict = config['orientation_dict']
    print(orientationDict['standard'])

    areas_dict = config['areas_dict']

    odf_calc = config['odf_calc']
    data_key = next(iter(dictionary))       #FIXME: hardcoded get the first key
    dictionary_wo_GT = copy.deepcopy(dictionary)
    GT_key = 'GT'
    dictionary_wo_GT.pop(GT_key)            # remove it

    for area in areas_dict:
        # given the indices from Matlab ArrShow
        orientation = areas_dict[area]['orientation']
        slice_ind = areas_dict[area]['slice_ind']
        matlab_area_TL = areas_dict[area]['matlab_area_TL'] #y,x
        matlab_area_DR = areas_dict[area]['matlab_area_DR']  #y,x
        directory = area + os.sep

        params = Parameters(orientation, slice_ind, matlab_area_TL, matlab_area_DR, directory, orientationDict=orientationDict, dictionary=dictionary)

        # Enables/disables interactive visualization and sf_calc (primary diffusion direction)
        i=0
        
        print(GT_key)
        data = load_data(GT_key, path_to_data, dictionary, odf_calc, orientationDict)
        # dictionary.pop(GT_key)            # remove it
        data_dict[GT_key]['data'] = data
        print('>> data loaded')
        print('>> Process GT first')
        data = data_dict[GT_key]['data']
        f = h5py.File('/home/vault/mfqb/mfqb102h/LASER_bipolar_revision/FOD/mask.h5', 'r')
        mask = f['mask'][:]
        f.close()
        try:
            assert mask.shape == tuple(config['N_x'], config['N_y'], config['N_z'], 1)
        except AssertionError as e:
            try:
                mask = mask.T
                assert mask.shape == tuple(config['N_x'], config['N_y'], config['N_z'], 1)
                print('mask was transposed')
                
            except AssertionError as e2:
                print("Assertion failed")
                raise

        # data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
        #             params.slicer_area_DR[0]:params.slicer_area_TL[0], 
        #             :]
        print(str(params.slicer_area_DR[1])+':'+str(params.slicer_area_TL[1])+', '+str(params.slicer_area_DR[0])+':'+str(params.slicer_area_TL[0]) + ', '+ 
                    str(params.slice_ind_low)+':'+str(params.slice_ind_high))
        GT_data = data*mask
        GT_data = GT_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                    params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                    params.slice_ind_low:params.slice_ind_high, :]
        print('GT_data shape:', GT_data.shape)
        org_shape = GT_data.shape
               

        # Start bootstrapping analysis TODO: parallelize over keys
        for data_key in dictionary_wo_GT:
            t = time()
            peaks_dirs_GT = peaks_dirs_GT.reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3], org_shape[4]))
            print(data_key)
            print('>> start bootstrapping')
            data1 = load_data_bootstrap(data_key, path_to_data, dictionary, odf_calc, orientationDict, n='1')
            data2 = load_data_bootstrap(data_key, path_to_data, dictionary, odf_calc, orientationDict, n='2')
            data_joint = np.stack((data1, data2), axis=-1)  
            
            angles = np.zeros((params.slicer_area_TL[1]-params.slicer_area_DR[1], params.slicer_area_TL[0]-params.slicer_area_DR[0], params.slice_ind_high-params.slice_ind_low, N_bootstrap, 2))

            peak_file_path =  params.directory +data_key + '_bootstrap_analysis'
            f = h5py.File(peak_file_path + '.h5', 'w')
            
            for bs in range(N_bootstrap):
                print('>> bootstrap number ', bs)
                bs_lookup = bost_vectors[bs,...]
                bs_data = data_joint[:,:,:,np.arange(len(bs_lookup)), bs_lookup]
                bs_data = bs_data*mask 
                bs_data = bs_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                                    params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                                    params.slice_ind_low:params.slice_ind_high, :]
                                
                save_dir = path_to_data + os.sep +'bootstraps' + os.sep + str(bs) + os.sep + data_key 
                create_directory(save_dir)
                save_path = save_dir + os.sep + 'data'
                data_path = convert_to_nii(data_joint, save_path)
                print('bs_data shape:', bs_data.shape)

                #TODO: MRtrix call
                proc = subprocess.Popen(["dwi2fod msmt_csd", data_path, path_to_data+'/wm_response.txt', save_dir+'_WM_FOD.nii', path_to_data+'/gm_response.txt',
                                         save_dir+'_GM.nii', path_to_data+'/csf_response.txt', save_dir+'_CSF.nii', '-grad '+ path_to_data+'/grad_126.b'])
                proc.wait()   # Blocks until it finish
                proc = subprocess.Popen(["sh2peaks -num 2 -threshold 0.05", save_dir+'_WM_FOD.nii', save_dir+'_PEAKS.nii'])
                proc.wait()   # Blocks until it finish

            #     peaks_dirs_bs_data = BS_data_csd_peaks.peak_dirs
            #     peak_values_bs_data = BS_data_csd_peaks.peak_values
            #     if N_bootstrap <= 10:
            #         f.create_dataset('peak_dirs_bs_' + str(bs), data=peaks_dirs_bs_data)
            #         f.create_dataset('peak_values_bs_' + str(bs), data=peak_values_bs_data)

            #     # Get rid of 3rd and higher peaks for simplicity
            #     peaks_dirs_bs_data = peaks_dirs_bs_data[...,0:2,:].reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3], org_shape[4]))
            #     peak_values_bs_data = peak_values_bs_data[...,0:2].reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3]))

            #     for pix in range(peaks_dirs_bs_data.shape[0]):
            #         angle_list = calc_angles(peaks_dirs_GT[pix,0,:], peaks_dirs_GT[pix,1,:], peaks_dirs_bs_data[pix,0,:], peaks_dirs_bs_data[pix,1,:])

            #         zer_dim = pix // (org_shape[1]*org_shape[2])
            #         fir_dim = (pix // org_shape[2]) % org_shape[1]
            #         sec_dim = pix % org_shape[2]

            #         angles[zer_dim, fir_dim, sec_dim, bs, :] = np.array(angle_list)
            #         peaks[zer_dim, fir_dim, sec_dim, bs, :] = peak_values_bs_data[pix,:]

            # print('>> bootstrapping done')
            # # Save angles and peak values
            # peaks_dirs_GT = peaks_dirs_GT.reshape((org_shape[0], org_shape[1], org_shape[2], org_shape[3], org_shape[4]))
            # peak_values_GT = peak_values_GT.reshape((org_shape[0], org_shape[1], org_shape[2], org_shape[3]))

            
            # f.create_dataset('angles', data=angles)
            # f.create_dataset('peaks', data=peaks)
            # f.create_dataset('org_peaks', data=peak_values_GT)
            # f.create_dataset('org_dirs', data=peaks_dirs_GT)
            # f.create_dataset('mask', data=mask)
            # f.close()
            print('>> saving done')
            print('Elapsed time: ', time()-t)



if __name__ == "__main__":
    main()

import os
import h5py
import numpy as np
import gc
import dipy.reconst.sfm as sfm
from dipy.reconst.odf import gfa
import nibabel as nib

import dipy.data as dpd
from dipy.data import get_sphere

import dipy.direction.peaks as dpp

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    auto_response_msmt,
    mask_for_response_msmt,
    multi_shell_fiber_response,
    response_from_mask_msmt,
)
import dipy.reconst.shm as shm
from dipy.segment.mask import median_otsu
from dipy.direction import peaks_from_model
from dipy.segment.tissue import TissueClassifierHMRF
from PIL import Image
import matplotlib.pyplot as plt
import copy
from dipy.core.sphere import unit_icosahedron
import cvxpy
from itertools import permutations, product
# use high angular resolution sphere
sphere = unit_icosahedron.subdivide(n=6)#get_sphere(name="symmetric724")


import yaml
from yaml import Loader
from time import time
import warnings
import subprocess

class Parameters:
    def __init__(self, orientation: str, 
                 slice_ind: tuple, 
                 matlab_area_TL: tuple, 
                 matlab_area_DR: tuple, 
                 directory: str,
                 orientationDict,
                 dictionary):
        self.orientation = orientation
        self.slice_ind_low = slice_ind[0]-1
        self.slice_ind_high = slice_ind[1]-1
        self.matlab_area_TL = matlab_area_TL
        self.matlab_area_DR = matlab_area_DR
        # translate coordinates from Matlab (origin bottom right) to the origins in the odf Slicer and for the rectangle drawing in the end
        self.slicer_area_TL, self.slicer_area_DR = self.coosMatlab2Slicer(orientationDict['standard'])
        self.rect_area_TL, self.rect_area_DR = self.coosMatlab2Rect(orientationDict['standard'])
        self.directory = directory
        self.dictionary = dictionary
        
        self.ensureDirectoryExists()

    
    def coosMatlab2Slicer(self, standardShape):
        N_y,N_x,N_z,N_q = standardShape
        # TL = x0,y0 DR = x1,y1
        x0 = self.matlab_area_TL[1]-1
        y0 = self.matlab_area_TL[0]-1
        x1 = self.matlab_area_DR[1]-1
        y1 = self.matlab_area_DR[0]-1
        
        if self.orientation == 'tra':
            #TODO: check if actually working
            areaTLNew = (y0+1, N_x - x0)
            areaDRNew = (y1, N_x - x1-1)
        elif self.orientation == 'cor':
            areaTLNew = (N_z-y1, N_x -x1-1)
            areaDRNew = (N_z-y0-1, N_x -x0)
        return areaTLNew, areaDRNew
    
    def coosMatlab2Rect(self, standardShape):
        N_y,N_x,N_z,N_q = standardShape
        # TL = x0,y0 DR = x1,y1
        x0 = self.matlab_area_TL[1]-1
        y0 = self.matlab_area_TL[0]-1
        x1 = self.matlab_area_DR[1]-1
        y1 = self.matlab_area_DR[0]-1
        if self.orientation == 'tra':
            #Additional border shift due to rectangle definition
            areaTLNew = (y0+1, N_x - x0)
            areaDRNew = (y1-1, N_x - x1-2)
        elif self.orientation == 'cor':
            areaTLNew = (N_z-y0-1, N_x -x0)
            areaDRNew = (N_z-y1, N_x -x1 - 1)
        return areaTLNew, areaDRNew
    
    def ensureDirectoryExists(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def createConfigTxt(self):
        completeName = os.path.join(self.directory, "parameters.txt")

        txtFile = open(completeName, "w")

        txtFile.write("Parameters used:\n\n")
        txtFile.write(Parameters.__str__(self))
        txtFile.close()

    def __str__(self) -> str:
        output_str = ''
        output_str += f'> used orientation: {self.orientation}\n'
        output_str += f'> used slice_ind: {self.slice_ind}\n'
        output_str += f'> used matlab_area_TL: {self.matlab_area_TL}\n'
        output_str += f'> used matlab_area_DR: {self.matlab_area_DR}\n'
        return output_str


def remove_black_borders(image_path, output_path, template=None, template_path=None):
    # Open the image
    if template:
        tmp = Image.open(template_path).convert("RGBA")
        img = Image.open(image_path).convert("RGBA")
    else:
        img = Image.open(image_path).convert("RGBA")
        tmp = img
    
    # Get the dimensions of the image
    width, height = img.size

    # Define the boundaries for cropping
    left, top, right, bottom = 0, 0, width, height

    # Scan from the top to find the first non-black row
    for y in range(height):
        row = tmp.crop((0, y, width, y + 1))
        if not all(pixel == (0, 0, 0, 255) for pixel in row.getdata()):
            top = y
            break

    # Scan from the bottom to find the last non-black row
    for y in range(height - 1, -1, -1):
        row = tmp.crop((0, y, width, y + 1))
        if not all(pixel == (0, 0, 0, 255) for pixel in row.getdata()):
            bottom = y + 1
            break

    # Scan from the left to find the first non-black column
    for x in range(width):
        column = tmp.crop((x, 0, x + 1, height))
        if not all(pixel == (0, 0, 0, 255) for pixel in column.getdata()):
            left = x
            break

    # Scan from the right to find the last non-black column
    for x in range(width - 1, -1, -1):
        column = tmp.crop((x, 0, x + 1, height))
        if not all(pixel == (0, 0, 0, 255) for pixel in column.getdata()):
            right = x + 1
            break
    
    # Crop the image using the found boundaries
    cropped_img = img.crop((left, top, right, bottom))
    
    # Save the cropped image
    cropped_img.save(output_path)

def normalize_path(path):
    """
    Normalize a file path to be platform-independent and usable in Python.
    
    Parameters:
        path (str): The file path as copied from Windows or Linux.
        
    Returns:
        str: The normalized path.
    """
    # Replace backslashes with forward slashes for initial consistency
    path = path.replace("\\", "/")
    # Use os.path.normpath to make it platform-specific
    normalized_path = os.path.normpath(path)

    if not os.path.splitext(normalized_path)[1]:  # No file extension
        # Append a trailing slash or backslash based on the OS
        if not normalized_path.endswith(os.sep):
            normalized_path += os.sep

    return normalized_path

def load_data(key, path_to_data, dictionary, odf_calc, orientationDict, file_format='hdf5'):
    if file_format == 'hdf5':
        dataFile = h5py.File(path_to_data + key +  os.sep + dictionary[key]+ '.h5', 'r')
        # if key == 'lpca':
        #     if odf_calc:
        #         data = abs(dataFile['DWI'][:])* 1000
        #     FA   = dataFile['FA_lam_0'][:]
        #     print('>> FA lpca shape:',FA.shape)
        data = abs(dataFile['DWI'][:].T)* 1000
        dataFile.close()

        # flip z-axis, because it is necessary from standard data shape #TODO: evaluate if necessary for tra and sag
        if odf_calc:
            
            try:
                print(data.shape)
                assert data.shape == tuple(orientationDict['standard'])
            except AssertionError as e:
                try:
                    data = data.T
                    assert data.shape == tuple(orientationDict['standard'])
                    print('data was transposed')
                    
                except AssertionError as e2:
                    print("Assertion failed")
                    raise
    elif file_format == 'nii':
        n1 = nib.load(path_to_data + key +  os.sep + dictionary[key]+ '.nii')
        data = n1.dataobj.get_unscaled()
        data = np.array(data)

    return data

def minmax_normalize(samples, out=None):
    """Min-max normalization of a function evaluated on the unit sphere

    Normalizes samples to ``(samples - min(samples)) / (max(samples) -
    min(samples))`` for each unit sphere.

    Parameters
    ----------
    samples : ndarray (..., N)
        N samples on a unit sphere for each point, stored along the last axis
        of the array.
    out : ndrray (..., N), optional
        An array to store the normalized samples.

    Returns
    -------
    out : ndarray, (..., N)
        Normalized samples.

    """
    if out is None:
        dtype = np.common_type(np.empty(0, 'float32'), samples)
        out = np.array(samples, dtype=dtype, copy=True)
    else:
        out[:] = samples

    sample_mins = np.min(samples, -1)[..., None]
    sample_maxes = np.max(samples, -1)[..., None]
    out -= sample_mins
    out = np.divide(out, (sample_maxes - sample_mins),
                        out=np.zeros_like(out),
                        where=out!=0)
    return out

def load_data_bootstrap(key, path_to_data, dictionary, odf_calc, orientationDict, n='1'):
    full_data_path = path_to_data + key +  os.sep + dictionary[key]
    full_data_path = full_data_path[:-1]
    # check for existence of file
    # if not os.path.isfile(full_data_path+'.h5'):
    #     raise FileNotFoundError(f"File {full_data_path+n+'.h5'} does not exist.")
    dataFile = h5py.File(full_data_path+n+'.h5', 'r')
    # if key == 'lpca':
    #     if odf_calc:
    #         data = abs(dataFile['DWI'][:])* 1000
    #     FA   = dataFile['FA_lam_0'][:]
    #     print('>> FA lpca shape:',FA.shape)
    data = abs(dataFile['DWI'][:].T)* 1000
    dataFile.close()

    # flip z-axis, because it is necessary from standard data shape #TODO: evaluate if necessary for tra and sag
    if odf_calc:
        
        try:
            # print(data.shape)
            assert data.shape == tuple(orientationDict['standard'])
        except AssertionError as e:
            try:
                data = data.T
                assert data.shape == tuple(orientationDict['standard'])
                print('data was transposed')
                
            except AssertionError as e2:
                print("Assertion failed")
                raise

    return data

def angle_between_vectors(a, b):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta)

from itertools import permutations, product

def angle_between_vectors(a, b):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta)

def calc_angles(GT_vector1, GT_vector2, PI_vector1, PI_vector2):
    ''' For entries where an entry is expected but none was fitte we return np.inf
        For entries where no entry is expected and none was fitted we return np.nan'''
    only1vecGT = False
    only1vecPI = False
    if np.all(np.isnan(GT_vector2)) and np.all(np.isnan(GT_vector1)):
        # print('Warning: no GT vectors!')
        return [np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    if np.all(np.isnan(PI_vector1)) and np.all(np.isnan(PI_vector2)):
        # print('Warning: no reco vectors!')
        # if vector 1 doesn't exist there can't be a vector 2
        if np.all(np.isnan(GT_vector2)) and not np.all(np.isnan(GT_vector1)):
            return [np.nan, np.inf], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
        else:
            return [np.inf, np.inf], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    
    if np.all(np.isnan(GT_vector2)):
        GT_vectors = [GT_vector1]   # Search for only one vector in the PI-vectors
        only1vecGT = True
        # print('Warning: only one GT vector!')
    else:
        GT_vectors = [GT_vector1, GT_vector2]

    if only1vecGT:
        PI_vectors = [PI_vector1]
        only1vecPI = True           #FIXME: In theory there is a case where two are fitted but only one GT vector exists
        # print('Warning: only one PI vector!')
    elif np.all(np.isnan(PI_vector2)):
        PI_vectors = [PI_vector1]
        # print('Warning: only one PI vector althoug 2 GT vectors!')
        only1vecPI = True
    else:
        PI_vectors = [PI_vector1, PI_vector2]

    # if only1vec:
    #     res_ang = [angle_between_vectors(GT_vector1, PI_vector1)]
    #     # if res_ang[0]>np.pi/2:
    #     #     res_ang[0] = np.pi - res_ang[0]
    # else:
    #     res_ang = [angle_between_vectors(GT_vector1, PI_vector1), angle_between_vectors(GT_vector2, PI_vector2)]
    #     # if res_ang[0]>np.pi/2:
    #     #     res_ang[0] = np.pi - res_ang[0]
    #     # if res_ang[1]>np.pi/2:
    #     #     res_ang[1] = np.pi - res_ang[1]
    best_combo = None
    smallest_ang = np.pi/2
    for pi_perm in permutations(PI_vectors):
        for signs in product([1, -1], repeat=2):
                angles = [angle_between_vectors(gt, s * pi) for gt, pi, s in zip(GT_vectors, pi_perm, signs)]
                if angles[0]>np.pi/2:
                    continue
                if not (only1vecGT or only1vecPI):
                    if angles[1]>np.pi/2:
                        continue
                if angles[0] < smallest_ang:
                    smallest_ang = angles[0]
                    best_combo = (pi_perm, signs, angles)             
                if not (only1vecGT or only1vecPI):
                    if angles[1] < smallest_ang:
                        smallest_ang = angles[1]
                        best_combo = (pi_perm, signs, angles)
    try:
        res_ang = best_combo[-1]
        
    except TypeError:
        print('No valid angle combination found!')
        print('GT vectors:', GT_vectors)
        print('PI vectors:', PI_vectors)
        res_ang = [np.inf, np.inf]
        associated_vec_1 = [np.nan, np.nan, np.nan]
        associated_vec_2 = [np.nan, np.nan, np.nan]
    if len(res_ang)==1 and only1vecGT:
        res_ang.append(np.nan)
        associated_vec_1 = best_combo[0][0] * best_combo[1][0]
        associated_vec_2 = [np.nan, np.nan, np.nan]
    elif len(res_ang)==1 and only1vecPI:
        res_ang.append(np.inf)
        associated_vec_1 = best_combo[0][0] * best_combo[1][0]
        associated_vec_2 = [np.nan, np.nan, np.nan]
    else:
        associated_vec_1 = best_combo[0][0] * best_combo[1][0]
        associated_vec_2 = best_combo[0][1] * best_combo[1][1]
    return np.array(res_ang), associated_vec_1, associated_vec_2

def convert_to_nii(data, save_path):
    data_abs = np.abs(data)
    data_path = save_path + '_abs.nii'
    affine=np.array([[-1.81818, 0, 0, 100.612],
                   [0, 1.81818, 0, -93.01261],
                   [0, 0, 3.75, -20.7228],
                   [0, 0, 0, 1]])

    img_abs = nib.Nifti1Image(data_abs, affine=affine)
    img_abs.header.set_sform(affine, code=1)
    img_abs.header.set_qform(affine, code=1)
    nib.save(img_abs, data_path)
    return data_path

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

def norm_vec_array(vector_array: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D array of vectors.

    Parameters:
    vector_array (np.ndarray): A 2D numpy array where each row represents a vector.

    Returns:
    np.ndarray: A 2D numpy array of the same shape with normalized vectors.
    """
    norms = np.linalg.norm(vector_array, axis=-1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    normalized_vectors = vector_array / norms
    return normalized_vectors

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    seq_path = normalize_path('..\\..\\..\\data\\raw\\1.0mm_126-dir_R3x3_dvs.h5')

    seqInfo = h5py.File(seq_path, 'r')
    bvals   = seqInfo['bvals'][:]
    bvecs   = seqInfo['bvecs'][:]
    seqInfo.close()

    bost_file = h5py.File('bootstrapping_vectors.h5', 'r')
    bost_vectors = bost_file['vectors'][:].squeeze()
    bost_file.close()

    N_bootstrap = config['N_bootstrap']
    num_fiber = config['num_fiber']
    bootstrap_dir_name = config['bootstrap_dir_name']


    print('Using N_bootstrap =', N_bootstrap)
    bost_vectors = bost_vectors[:N_bootstrap,...]

    gtab = gradient_table(bvals, bvecs, atol=3e-2)

    path_to_data = config['path_to_data']
    path_to_data = normalize_path(path_to_data)
    path_to_latent = config['path_to_latent']
    path_to_latent = normalize_path(path_to_latent)
    dictionary = config['dictionary']
    data_dict = {key: {'data':None,'FA':None} for key in dictionary}
    # dictionary = {'muse': 'MuseRecon_combined_slices', 'llr': 'JETS2', 'dec':'DecRecon_combined_slices','DTI': 'DecRecon_combined_slices'}

    orientationDict = config['orientation_dict']
    print(orientationDict['standard'])

    areas_dict = config['areas_dict']

    odf_calc = config['odf_calc']
    data_key = next(iter(dictionary))       #FIXME: hardcoded get the first key
    dictionary_wo_GT = copy.deepcopy(dictionary)
    GT_key = 'GT'
    dictionary_wo_GT.pop(GT_key)            # remove it

    for area in areas_dict:
        # given the indices from Matlab ArrShow
        orientation = areas_dict[area]['orientation']
        slice_ind = areas_dict[area]['slice_ind']
        matlab_area_TL = areas_dict[area]['matlab_area_TL'] #y,x
        matlab_area_DR = areas_dict[area]['matlab_area_DR']  #y,x
        directory = area + os.sep

        params = Parameters(orientation, slice_ind, matlab_area_TL, matlab_area_DR, directory, orientationDict=orientationDict, dictionary=dictionary)

        # Enables/disables interactive visualization and sf_calc (primary diffusion direction)
        i=0
        
        print(GT_key)
        data = load_data(GT_key, path_to_data, dictionary, odf_calc, orientationDict, file_format='nii')
        data_dict[GT_key]['data'] = data
        print('>> data loaded')
        f = h5py.File(path_to_data + 'mask.h5', 'r')
        mask = f['mask'][:]
        f.close()
        try:
            assert mask.shape == tuple(orientationDict['standard_mask'])
        except AssertionError as e:
            try:
                mask = mask.T
                assert mask.shape == tuple(orientationDict['standard_mask'])
                print('mask was transposed')
                
            except AssertionError as e2:
                print("Assertion failed")
                raise

        # data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
        #             params.slicer_area_DR[0]:params.slicer_area_TL[0], 
        #             :]
        print(str(params.slicer_area_DR[1])+':'+str(params.slicer_area_TL[1])+', '+str(params.slicer_area_DR[0])+':'+str(params.slicer_area_TL[0]) + ', '+ 
                    str(params.slice_ind_low)+':'+str(params.slice_ind_high))
        GT_data = data.copy()
        GT_data = GT_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                            params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                            params.slice_ind_low:params.slice_ind_high, :]
        print('GT_data shape:', GT_data.shape)
        org_shape = GT_data.shape

        # Normalize_vectors
        GT_vec_1 = GT_data[...,0:3]
        GT_vec_1_norm = norm_vec_array(GT_vec_1)
        GT_vec_1_norm_flat = GT_vec_1_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))
        if num_fiber==2:
            GT_vec_2 = GT_data[...,3:6]
            GT_vec_2_norm = norm_vec_array(GT_vec_2)
            GT_vec_2_norm_flat = GT_vec_2_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))

               

        # Start bootstrapping analysis TODO: parallelize over keys
        for data_key in dictionary_wo_GT:
            t = time()
            # peaks_dirs_GT = peaks_dirs_GT.reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3], org_shape[4]))
            print(data_key)
            print('>> start bootstrapping')
            data1 = load_data_bootstrap(data_key, path_to_latent, dictionary, odf_calc, orientationDict, n='1')
            data2 = load_data_bootstrap(data_key, path_to_latent, dictionary, odf_calc, orientationDict, n='2')
            data_joint = np.stack((data1, data2), axis=-1)  
            
            angles = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, num_fiber))
            vec1_comb = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, 3))
            if num_fiber==2:
                vec2_comb = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, 3))
            
            for bs in range(N_bootstrap):
                print('>> bootstrap number ', bs)
                bs_lookup = bost_vectors[bs,...]
                bs_data = data_joint[:,:,:,np.arange(len(bs_lookup)), bs_lookup]
                bs_data = bs_data*mask 
                bs_data = bs_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                                    params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                                    params.slice_ind_low:params.slice_ind_high, :]
                                
                save_dir = path_to_data + bootstrap_dir_name + os.sep + str(bs) + os.sep + data_key 
                create_directory(save_dir)
                save_path = save_dir + os.sep + 'data'
                print('bs_data shape:', bs_data.shape)
                data_path = convert_to_nii(bs_data, save_path)
                res_dict = save_dir + os.sep + data_key

                print('data saved to:', data_path)
                print('res dict:', res_dict)

                #TODO: MRtrix call
                proc = subprocess.Popen([
                                        "dwi2fod", "msmt_csd",
                                        data_path,
                                        path_to_data + "/wm_response.txt", res_dict + "_WM_FOD.nii",
                                        path_to_data + "/gm_response.txt",res_dict + "_GM.nii",
                                        path_to_data + "/csf_response.txt",res_dict + "_CSF.nii",
                                        "-grad", path_to_data + "/grad_126.b" , "-force"
                                        ])
                proc.wait()   # Blocks until it finish
                proc = subprocess.Popen(["sh2peaks", "-num", str(num_fiber), res_dict+'_WM_FOD.nii', res_dict+'_PEAKS.nii', '-force'])
                proc.wait()   # Blocks until it finish

                vector_data_file = nib.load(res_dict+'_PEAKS.nii')
                vector_data = vector_data_file.dataobj.get_unscaled()
                vector_data = np.array(vector_data)

                rec_vec_1 = vector_data[...,0:3]
                rec_vec_1_norm = norm_vec_array(rec_vec_1)
                if num_fiber==2:
                    rec_vec_2 = vector_data[...,3:6]
                    rec_vec_2_norm = norm_vec_array(rec_vec_2)

                rec_vec_1_norm_flat = rec_vec_1_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))
                if num_fiber==2:
                    rec_vec_2_norm_flat = rec_vec_2_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))

                for pix in range(rec_vec_1_norm_flat.shape[0]):
                    if num_fiber==1:
                        angles_diff = angle_between_vectors(GT_vec_1_norm_flat[pix,:], rec_vec_1_norm_flat[pix,:])
                        asso_vec_1 = rec_vec_1_norm_flat[pix,:]
                    elif num_fiber==2:
                        angles_diff, asso_vec_1, asso_vec_2 = calc_angles(GT_vec_1_norm_flat[pix,:], GT_vec_2_norm_flat[pix,:], rec_vec_1_norm_flat[pix,:], rec_vec_2_norm_flat[pix,:])

                    zer_dim = pix // (org_shape[1]*org_shape[2])
                    fir_dim = (pix // org_shape[2]) % org_shape[1]
                    sec_dim = pix % org_shape[2]

                    angles[zer_dim, fir_dim, sec_dim, bs, :] = np.array(angles_diff)
                    vec1_comb[zer_dim, fir_dim, sec_dim,bs,:] = asso_vec_1
                    if num_fiber==2:
                        vec2_comb[zer_dim, fir_dim, sec_dim,bs,:] = asso_vec_2

                # remove created directory to safe space
                if config['remove_bootstrap_data']:
                    proc = subprocess.Popen(["rm", save_dir, '-r'])
                    proc.wait()   # Blocks until it finish

            print('>> bootstrapping done')
            # Save angles and peak values
            angle_error_path =  path_to_data + bootstrap_dir_name + os.sep + 'bootstrap_analysis_' + data_key
            f = h5py.File(angle_error_path + '.h5', 'w')            
            f.create_dataset('angles', data=angles)
            f.create_dataset('org_vec_1', data=GT_vec_1_norm)
            f.create_dataset('vec1_comb', data=vec1_comb)
            if num_fiber==2:
                f.create_dataset('org_vec_2', data=GT_vec_2_norm)
                f.create_dataset('vec2_comb', data=vec2_comb)           
            f.close()
            print('>> saving done')
            print('Elapsed time: ', time()-t)



if __name__ == "__main__":
    main()
