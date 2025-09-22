import matplotlib.pyplot as plt
import numpy as np

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_fnames, get_sphere
from dipy.denoise.localpca import mppca
import dipy.direction.peaks as dp
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    auto_response_msmt,
    mask_for_response_msmt,
    multi_shell_fiber_response,
    response_from_mask_msmt,
)
import dipy.reconst.shm as shm
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.viz import actor, window
import h5py
from dipy.direction import peaks_from_model

import cvxpy
from itertools import permutations, product

sphere = get_sphere(name="symmetric724")


"""
.. _reconst_sfm:

==============================================
Reconstruction with the Sparse Fascicle Model
==============================================

In this example, we will use the Sparse Fascicle Model (SFM) [Rokem2015]_, to
reconstruct the fiber Orientation Distribution Function (fODF) in every voxel.

First, we import the modules we will use in this example:
"""

import os
import h5py
import numpy as np
import gc
import dipy.reconst.sfm as sfm
from dipy.reconst.odf import gfa

import dipy.data as dpd
import dipy.direction.peaks as dpp
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.viz import window, actor
from PIL import Image
import matplotlib.pyplot as plt
import copy

import yaml
from yaml import Loader
from time import time

class Parameters:
    def __init__(self, orientation: str, 
                 slice_ind: int, 
                 matlab_area_TL: tuple, 
                 matlab_area_DR: tuple, 
                 directory: str,
                 orientationDict,
                 dictionary):
        self.orientation = orientation
        self.slice_ind = slice_ind-1
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

def angle_between_vectors(a, b):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta)

def calc_angles(GT_vector1, GT_vector2, PI_vector1, PI_vector2):

    only1vec = False
    if np.array_equal(GT_vector2, np.array([0,0,0])) and np.array_equal(GT_vector1, np.array([0,0,0])):
        print('Warning: no GT vectors!')
        return [180, 180]
    if np.array_equal(PI_vector1, np.array([0,0,0])) and np.array_equal(PI_vector2, np.array([0,0,0])):
        print('Warning: no reco vectors!')
        return [np.nan, np.nan]
    
    if np.array_equal(GT_vector2, np.array([0,0,0])):
        GT_vectors = [GT_vector1]   # Search for only one vector in the PI-vectors
        only1vec = True
        # print('Warning: only one GT vector!')
    else:
        GT_vectors = [GT_vector1, GT_vector2]

    if only1vec:
        PI_vectors = [PI_vector1]
        # print('Warning: only one PI vector!')
    elif np.array_equal(PI_vector2, np.array([0,0,0])):
        PI_vectors = [PI_vector1]
        print('Warning: only one PI vector althoug 2 GT vectors!')
    else:
        PI_vectors = [PI_vector1, PI_vector2]


    best_combo = None
    best_total_angle = float("inf")

    # Try all PI permutations and sign flips
    for pi_perm in permutations(PI_vectors):
        for signs in product([1, -1], repeat=2):
            angles = [angle_between_vectors(gt, s * pi) 
                    for gt, pi, s in zip(GT_vectors, pi_perm, signs)]
            total_angle = sum(angles)
            if total_angle < best_total_angle:
                best_total_angle = total_angle
                best_combo = (pi_perm, signs, angles)
    # print(best_combo)
    res_ang = best_combo[-1]
    if len(res_ang)==1:
        res_ang.append(np.nan)
    return res_ang


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

        key = next(iter(dictionary))   # get the first key
        
        print(key)
        data = load_data(key, path_to_data, dictionary, odf_calc, orientationDict)
        dictionary.pop(key)            # remove it
        data_dict[key]['data'] = data
        print('>> data loaded')
        print('>> Process GT first')
        data = data_dict[key]['data']
        if params.orientation == 'cor': #TODO: don't flip but adjust camera view
            data = np.flip(data, 2)
        
        b0_mask, mask = median_otsu(data, median_radius=10, numpass=1, vol_idx=[0, 1])   #TODO: hardcoded???

        # data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
        #             params.slicer_area_DR[0]:params.slicer_area_TL[0], 
        #             :]
        print(str(params.slicer_area_DR[1])+':'+str(params.slicer_area_TL[1])+', '+str(params.slicer_area_DR[0])+':'+str(params.slicer_area_TL[0]))
        auto_response_wm, auto_response_gm, auto_response_csf = auto_response_msmt(
            gtab, b0_mask, roi_radii=20
        )
        print(b0_mask.shape)
        ubvals = unique_bvals_tolerance(gtab.bvals)
        response_mcsd = multi_shell_fiber_response(
            sh_order_max=SH_order, 
            bvals=ubvals,
            wm_rf=auto_response_wm,
            gm_rf=auto_response_gm,
            csf_rf=auto_response_csf,
        )

        

        # print('Response output: ', auto_response_wm)
        # print('Array entries 1 and 2 should be identical: ', auto_response_wm[0][1] == auto_response_wm[0][2])
        # print('Array entry 0 should be around 5 times larger than 1 and 2: ', auto_response_wm[0][0] >= auto_response_wm[0][1]*5)
        # if not (auto_response_wm[0][0] >= auto_response_wm[0][1]*5):
        #     print('WARNING: only a ratio of ', auto_response_wm[0][0]/auto_response_wm[0][1], ' between axial and radial diffusivity')
        #     print('Change roi_radii or fa threshold in auto_response_ssst')

        

        mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)

        ###############################################################################
        # We can extract the peaks from the ODF, and plot these as well
        data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                    params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                    params.slice_ind:params.slice_ind+1]
        mask = mask[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                    params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                    params.slice_ind:params.slice_ind+1]
        
        print('>> peak extraction')
        GT_csd_peaks = peaks_from_model(
        model=mcsd_model,
        data=data,
        sphere=sphere,
        relative_peak_threshold=rel_peak_threshold,    #TODO: hardcoded???
        min_separation_angle=min_separation_angle,        #TODO: hardcoded???
        normalize_peaks=False,
        mask=mask,
        parallel=True,
        num_processes=2,               #TODO: hardcoded???
        )

        ap = shm.anisotropic_power(GT_csd_peaks.shm_coeff)
        beta = 0.4
        nclass = 3
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)

        csf = np.where(final_segmentation == 1, 1, 0)
        gm = np.where(final_segmentation == 2, 1, 0)
        wm_mask = np.where(final_segmentation == 3, 1, 0)
        
        


        print('>> peak extraction done')
        peaks_dirs_GT = GT_csd_peaks.peak_dirs
        peak_values_GT = GT_csd_peaks.peak_values

        # Get rid of 3rd and higher peaks for simplicity
        peaks_dirs_GT = peaks_dirs_GT[...,0:2,:] * wm_mask[...,None, None]
        org_shape = peaks_dirs_GT.shape
        
        
        peak_values_GT = peak_values_GT[...,0:2] * wm_mask[...,None]
        

        # Start bootstrapping analysis
        for key in dictionary:
            t = time()
            peaks_dirs_GT = peaks_dirs_GT.reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3], org_shape[4]))
            peak_values_GT = peak_values_GT.reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3]))
            print(key)
            print('>> start bootstrapping')
            data1 = load_data_bootstrap(key, path_to_data, dictionary, odf_calc, orientationDict, n='1')
            data2 = load_data_bootstrap(key, path_to_data, dictionary, odf_calc, orientationDict, n='2')

            data_joint = np.stack((data1, data2), axis=-1)  
            angles = np.zeros((params.slicer_area_TL[1]-params.slicer_area_DR[1], params.slicer_area_TL[0]-params.slicer_area_DR[0], 1, N_bootstrap, 2))
            peaks = np.zeros((params.slicer_area_TL[1]-params.slicer_area_DR[1], params.slicer_area_TL[0]-params.slicer_area_DR[0], 1, N_bootstrap, 2))
            for bs in range(N_bootstrap):
                print('>> bootstrap number ', bs)
                bs_lookup = bost_vectors[bs,...]
                bs_data = data_joint[:,:,:,np.arange(len(bs_lookup)), bs_lookup]


                if params.orientation == 'cor': #TODO: don't flip but adjust camera view
                    bs_data = np.flip(bs_data, 2)
                
                bs_data = bs_data*mask[...,None]
                # bs_data = bs_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                #             params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                #             :]
                print(str(params.slicer_area_DR[1])+':'+str(params.slicer_area_TL[1])+', '+str(params.slicer_area_DR[0])+':'+str(params.slicer_area_TL[0]))
                # auto_response_wm, auto_response_gm, auto_response_csf = auto_response_msmt(                           TODO: NOT SURE IF IT IS VALID TO JUST USE THE SAME RESPONSE ALL THE TIME
                #     gtab, bs_data, roi_radii=3
                # )
                print(bs_data.shape)
                ubvals = unique_bvals_tolerance(gtab.bvals)
                response_mcsd = multi_shell_fiber_response(
                    sh_order_max=SH_order, 
                    bvals=ubvals,
                    wm_rf=auto_response_wm,
                    gm_rf=auto_response_gm,
                    csf_rf=auto_response_csf,
                )

                print('Response output: ', auto_response_wm)
                print('Array entries 1 and 2 should be identical: ', auto_response_wm[0][1] == auto_response_wm[0][2])
                print('Array entry 0 should be around 5 times larger than 1 and 2: ', auto_response_wm[0][0] >= auto_response_wm[0][1]*5)
                if not (auto_response_wm[0][0] >= auto_response_wm[0][1]*5):
                    print('WARNING: only a ratio of ', auto_response_wm[0][0]/auto_response_wm[0][1], ' between axial and radial diffusivity')
                    print('Change roi_radii or fa threshold in auto_response_ssst')

                

                mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)

                ###############################################################################
                # We can extract the peaks from the ODF, and plot these as well
                bs_data = bs_data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                            params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                            params.slice_ind:params.slice_ind+1]
                print('>> peak extraction')
                BS_data_csd_peaks = peaks_from_model(
                model=mcsd_model,
                data=bs_data,
                sphere=sphere,
                relative_peak_threshold=rel_peak_threshold,
                min_separation_angle=min_separation_angle,
                normalize_peaks=False,
                mask=mask,
                parallel=True,
                num_processes=2,               #TODO: hardcoded???
                )
                print('>> peak extraction done')

                peaks_dirs_bs_data = BS_data_csd_peaks.peak_dirs
                peak_values_bs_data = BS_data_csd_peaks.peak_values

                # Get rid of 3rd and higher peaks for simplicity
                peaks_dirs_bs_data = peaks_dirs_bs_data[...,0:2,:].reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3], org_shape[4]))
                peak_values_bs_data = peak_values_bs_data[...,0:2].reshape((org_shape[0]* org_shape[1] * org_shape[2], org_shape[3]))

                for pix in range(peaks_dirs_bs_data.shape[0]):
                    angle_list = calc_angles(peaks_dirs_GT[pix,0,:], peaks_dirs_GT[pix,1,:], peaks_dirs_bs_data[pix,0,:], peaks_dirs_bs_data[pix,1,:])

                    zer_dim = pix // (org_shape[1]*org_shape[2])
                    fir_dim = (pix // org_shape[2]) % org_shape[1]
                    sec_dim = pix % org_shape[2]

                    angles[zer_dim, fir_dim, sec_dim, bs, :] = np.array(angle_list)
                    peaks[zer_dim, fir_dim, sec_dim, bs, :] = peak_values_bs_data[pix,:]

            print('>> bootstrapping done')
            # Save angles and peak values
            peaks_dirs_GT = peaks_dirs_GT.reshape((org_shape[0], org_shape[1], org_shape[2], org_shape[3], org_shape[4]))
            peak_values_GT = peak_values_GT.reshape((org_shape[0], org_shape[1], org_shape[2], org_shape[3]))

            peak_file_path =  params.directory +key + '_bootstrap_analysis'
            f = h5py.File(peak_file_path + '.h5', 'w')
            f.create_dataset('angles', data=angles)
            f.create_dataset('peaks', data=peaks)
            f.create_dataset('org_peaks', data=peak_values_GT)
            f.create_dataset('org_dirs', data=peaks_dirs_GT)
            f.create_dataset('wm_mask', data=wm_mask)
            f.create_dataset('mask', data=mask)
            f.close()
            print('>> saving done')
            print('Elapsed time: ', time()-t)



if __name__ == "__main__":
    main()
