"""
This module implements a bootstrapping analysis for diffusion MRI data using MRtrix3.

Authors:
    Julius Glaser <julius-glaser@gmx.de>
"""

import os
import h5py
import numpy as np
import nibabel as nib
import yaml
from yaml import Loader
from time import time
import warnings
import subprocess
from PIL import Image
import copy
from itertools import permutations, product


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

def normalize_path(path: str) -> str:
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

def load_data(key: str, path_to_data: str, dictionary: dict, file_format: str = 'hdf5') -> np.ndarray:
    """
    load diffusion data and bring to adequate shape.
    
    Parameters:
        key (str): The file path as copied from Windows or Linux.
        path_to_data (str): The base path to the data.
        dictionary (dict): A dictionary mapping keys to file names.
        file_format (str): The format of the file to load ('hdf5' or 'nii').
        
    Returns:
        np.ndarray: The loaded and processed diffusion data.
    """

    if file_format == 'hdf5':
        dataFile = h5py.File(path_to_data + key +  os.sep + dictionary[key]+ '.h5', 'r')
        data = abs(dataFile['DWI'][:].T)* 1000
        dataFile.close()

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

def load_data_bootstrap(key: str, path_to_data: str, dictionary: dict, expected_shape: tuple, n: str = '1') -> np.ndarray:
    """
    load diffusion data and bring to adequate shape.
    
    Parameters:
        key (str): The file path as copied from Windows or Linux.
        path_to_data (str): The base path to the data.
        dictionary (dict): A dictionary mapping keys to file names.
        expected_shape (tuple): expected shapes for calculations.
        n (str): The number of the bootstrap sample.

    Returns:
        np.ndarray: The loaded and processed diffusion data.
    """
    full_data_path = path_to_data + key +  os.sep + dictionary[key]
    full_data_path = full_data_path[:-1]
    
    dataFile = h5py.File(full_data_path+n+'.h5', 'r')
    data = abs(dataFile['DWI'][:].T)* 1000
    data = ensure_shape(data, tuple(expected_shape))
    dataFile.close()

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

def ensure_shape(arr: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """
    Ensure that arr has expected_shape.
    If not, try transposing it.
    Raise ValueError if neither works.
    """
    if arr.shape == expected_shape:
        return arr

    transposed = arr.T
    if transposed.shape == expected_shape:
        return transposed

    raise ValueError(
        f"Array has shape {arr.shape}, "
        f"transpose has shape {transposed.shape}, "
        f"expected {expected_shape}"
    )

def main():

    # load config and set variables
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    N_bootstrap = config['N_bootstrap']
    num_fiber = config['num_fiber']
    bootstrap_dir_name = config['bootstrap_dir_name']

    path_to_data = config['path_to_data']
    path_to_data = normalize_path(path_to_data)
    path_to_latent = config['path_to_latent']
    path_to_latent = normalize_path(path_to_latent)
    dictionary = config['dictionary']

    orientationDict = config['orientation_dict']
    print(orientationDict['standard'])

    slice_ind = config['slice_ind']

    # load bootstrapping vectors, which indicate which dataset to use for each bootstrap
    bost_file = h5py.File('bootstrapping_vectors.h5', 'r')
    bost_vectors = bost_file['vectors'][:].squeeze()
    bost_file.close()

    print('Using N_bootstrap =', N_bootstrap)
    bost_vectors = bost_vectors[:N_bootstrap,...]

    # Load ground truth data
    data_key = next(iter(dictionary))
    dictionary_wo_GT = copy.deepcopy(dictionary)
    GT_key = 'GT'
    dictionary_wo_GT.pop(GT_key)            # remove it
    
    data = load_data(GT_key, path_to_data, dictionary, file_format='nii')
    print('>> data loaded')

    GT_data = data.copy()
    GT_data = GT_data[:, :, slice_ind[0]:slice_ind[1], :]
    print('GT_data shape:', GT_data.shape)
    org_shape = GT_data.shape

    # Normalize_vectors and safe for angle difference calculation
    GT_vec_1 = GT_data[...,0:3]
    GT_vec_1_norm = norm_vec_array(GT_vec_1)
    GT_vec_1_norm_flat = GT_vec_1_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))
    if num_fiber==2:
        GT_vec_2 = GT_data[...,3:6]
        GT_vec_2_norm = norm_vec_array(GT_vec_2)
        GT_vec_2_norm_flat = GT_vec_2_norm.reshape((org_shape[0]* org_shape[1] * org_shape[2], 3))

    # load mask
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

    # Start bootstrapping analysis TODO: parallelize over keys
    for data_key in dictionary_wo_GT:
        t = time()
        print(data_key)
        print('>> start bootstrapping')

        # load the datasets and stack them
        data1 = load_data_bootstrap(data_key, path_to_latent, dictionary, orientationDict['standard'], n='1')
        data2 = load_data_bootstrap(data_key, path_to_latent, dictionary, orientationDict['standard'], n='2')
        data_joint = np.stack((data1, data2), axis=-1)  
        
        # Prepare arrays to store results
        angles = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, num_fiber))
        vec1_comb = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, 3))
        if num_fiber==2:
            vec2_comb = np.zeros((org_shape[0], org_shape[1], org_shape[2], N_bootstrap, 3))
        
        # run bootstrapping
        for bs in range(N_bootstrap):
            print('>> bootstrap number ', bs)
            bs_lookup = bost_vectors[bs,...]
            # select the combination of the first and second dataset according to the bootstrapping vectors
            bs_data = data_joint[:,:,:,np.arange(len(bs_lookup)), bs_lookup]
            bs_data = bs_data*mask 
            bs_data = bs_data[:, :, slice_ind[0]:slice_ind[1], :]

            # save bs data of this sample as nii for mrtrix processing                
            save_dir = path_to_data + bootstrap_dir_name + os.sep + str(bs) + os.sep + data_key 
            create_directory(save_dir)
            save_path = save_dir + os.sep + 'data'
            print('bs_data shape:', bs_data.shape)
            data_path = convert_to_nii(bs_data, save_path)
            res_dict = save_dir + os.sep + data_key

            print('data saved to:', data_path)
            print('res dict:', res_dict)

            # MRtrix call
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

            # load peak data and calculate angle differences
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

            # calculate angle differences for all voxels
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

        # Save difference angles and peak values
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
