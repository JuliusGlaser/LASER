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

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    seq_path = normalize_path('..\\..\\..\\data\\raw\\1.0mm_126-dir_R3x3_dvs.h5')

    seqInfo = h5py.File(seq_path, 'r')
    bvals   = seqInfo['bvals'][:]
    bvecs   = seqInfo['bvecs'][:]
    seqInfo.close()

    gtab = gradient_table(bvals, bvecs, atol=3e-2)

    path_to_data = config['path_to_data']
    path_to_data = normalize_path(path_to_data)
    dictionary = config['dictionary']
    data_dict = {key: {'data':None,'FA':None} for key in dictionary}
    # dictionary = {'muse': 'MuseRecon_combined_slices', 'llr': 'JETS2', 'dec':'DecRecon_combined_slices','DTI': 'DecRecon_combined_slices'}

    orientationDict = config['orientation_dict']
    print(orientationDict['standard'])

    areas_dict = config['areas_dict']

    interactive = config['interactive']
    sf_calc = config['sf_calc']
    odf_calc = config['odf_calc']
    plot_joint_fig = config['plot_joint_fig']
                         
    for key in dictionary:
        print(key)
        print('>> load data')
        data = load_data(key, path_to_data, dictionary, odf_calc, orientationDict)
        data_dict[key]['data'] = data
        print('>> data loaded')

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
            
        for key in dictionary:
            print(key)
            print('>> start processing')
            data = data_dict[key]['data']
            if params.orientation == 'cor': #TODO: don't flip but adjust camera view
                data = np.flip(data, 2)
            
            b0_mask, mask = median_otsu(data, median_radius=4, numpass=1, vol_idx=[0, 1])   #TODO: hardcoded???
            data = data*b0_mask
            # data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
            #             params.slicer_area_DR[0]:params.slicer_area_TL[0], 
            #             :]
            print(str(params.slicer_area_DR[1])+':'+str(params.slicer_area_TL[1])+', '+str(params.slicer_area_DR[0])+':'+str(params.slicer_area_TL[0]))
            auto_response_wm, auto_response_gm, auto_response_csf = auto_response_msmt(
                gtab, data, roi_radii=3
            )
            print(data.shape)
            ubvals = unique_bvals_tolerance(gtab.bvals)
            response_mcsd = multi_shell_fiber_response(
                sh_order_max=8, #TODO: hardcoded???
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
            data = data[params.slicer_area_DR[1]:params.slicer_area_TL[1],
                        params.slicer_area_DR[0]:params.slicer_area_TL[0], 
                        :]
            if True: #TODO: add comments
                print('>> peak extraction')
                csd_peaks = peaks_from_model(
                model=mcsd_model,
                data=data,
                sphere=sphere,
                relative_peak_threshold=0.1,    #TODO: hardcoded???
                min_separation_angle=25,        #TODO: hardcoded???
                normalize_peaks=False,
                parallel=True,
                num_processes=2,               #TODO: hardcoded???
                )

                peak_dirs = csd_peaks.peak_dirs
                peak_values = csd_peaks.peak_values
                gfa = csd_peaks.gfa

                peak_file_path =  params.directory +key + '_peak_extraction'
                f = h5py.File(peak_file_path + '.h5', 'w')
                f.create_dataset('peak_dirs', data=peak_dirs)
                f.create_dataset('peak_values', data=peak_values)
                f.create_dataset('gfa', data=gfa)
                f.create_dataset('wm_response', data=auto_response_wm)
                f.create_dataset('gm_response', data=auto_response_gm)
                f.create_dataset('csf_response', data=auto_response_csf)
                f.create_dataset('b0_mask', data=b0_mask)
                f.close()
                print('>> peak extraction done')


                
            i= i+1

            
        params.createConfigTxt()

if __name__ == "__main__":
    main()