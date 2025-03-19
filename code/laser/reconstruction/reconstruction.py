"""
This module implements and uses several reconstructions:
- MUSE
- LASER
- AE regularized MUSE
- denoising of MUSE data

Authors:
    Julius Glaser <julius-glaser@gmx.de>
"""


import argparse
import h5py
import os

import numpy as np
import sigpy as sp

from sigpy.mri import retro, app, sms, muse
from os.path import exists

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from laser.training.models.nn import autoencoder as ae

import yaml
from yaml import Loader
from time import time

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

def get_sms_phase_shift_torch(ishape:tuple, MB:int, yshift:list, device:str)->torch.Tensor:

    """
    Args:
        ishape (tuple or list): input shape of [..., Nz, Ny, Nx].
        MB (int): multi-band factor.
        yshift (tuple or list): use custom yshift.

    References:
        * Breuer FA, Blaimer M, Heidemann RM, Mueller MF,
            Griswold MA, Jakob PM.
            Controlled aliasing in parallel imagin results in
            higher acceleration (CAIPIRINHA) for multi-slice imaging.
            Magn. Reson. Med. 53:684-691 (2005).
    """
    Nz, Ny, Nx = ishape[-3:]

    phi = torch.ones(ishape, dtype=torch.complex64).to(device)
    phi.requires_grad = False
    bas = 2 * torch.pi / 2
    bas = torch.tensor(bas).to(device)
    bas.requires_grad = False
    if yshift is None:
        yshift = (torch.arange(Nz)) / MB
    else:
        assert (len(yshift) == Nz)

    print(' > sms: yshift ', yshift)

    lx = torch.arange(Nx) - Nx // 2
    lx.requires_grad = False
    ly = torch.arange(Ny) - Ny // 2
    ly.requires_grad = False

    mx, my = torch.meshgrid(lx, ly, indexing='xy')
    my.requires_grad = False
    my = my.to(device)
    for z in range(Nz):
        slice_yshift = (bas * yshift[z]).to(torch.float64).to(device)
        slice_yshift.requires_grad = False
        phi[ z, :, :] = torch.exp(1j * my * slice_yshift)

    return phi

def add_noise(x_clean, scale, noiseType = 'gaussian'):

    if noiseType== 'gaussian':
        x_noisy = x_clean + np.random.normal(loc = 0,
                                            scale = scale,
                                            size=x_clean.shape)
    elif noiseType== 'rician':
        noise1 =np.random.normal(0, scale, size=x_clean.shape)
        noise2 = np.random.normal(0, scale, size=x_clean.shape)
        x_noisy = np.sqrt((x_clean + noise1) ** 2 + noise2 ** 2)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy

def Decoder_for(model:torch.nn.Module, N_x:int, N_y:int, N_z:int, Q:int, b0:torch.Tensor, x_1:torch.Tensor)-> torch.Tensor:
    '''
    Takes an input data through the decoder trained for compressing signal evolution
    Args:
        model (torch.nn.Module): the neural network used for decoding,
        N_x (int): Size of x-dimension,
        N_y (int): Size of y-dimension,
        N_z (int): Size of z-dimension,
        Q (int): Size of q-dimension,
        b0 (torch.Tensor): b0 to be used for scaling of AE output, shape (X*Y*Z,1),
        x_1 (torch.Tensor): latent real-valued reconstructions of DWI data, shape (X*Y, L)
    Returns:
        torch.Tensor: decoded, b0 scaled, complex valued, reshaped DWI reconstructions, shape (Q, 1, 1, Z, X, Y)

    L = number of latent variables
    X = number of frequency columns
    Y = number of phase-encoding lines
    Z = number of acquired slices
    Q = number of diffusion directions
    '''
    out = model.decode(x_1)
    out = (out*torch.real(b0) + 1j*torch.imag(b0)*out).reshape(N_z,N_x, N_y,Q)
    out_scaled = out.permute(-1, 0, 1, 2)    #Q,Z,X,Y,2
    out_scaled = torch.reshape(out_scaled, (Q,1,1,N_z,N_x,N_y))

    return out_scaled

def Multi_shot_for(x:torch.Tensor, phase:torch.Tensor)->torch.Tensor:
    '''
    Splits the acquired data in seperate acquired shots according to the shot phases
    Args:
        x (torch.Tensor): DWI data, shape (Q, 1, 1, Z, X, Y)
        phase (torch.Tensor): shot phases, shape (1, S, 1, Z, X, Y)
    Returns:
        torch.Tensor: shot split diffusion signal, shape (Q, S, 1, Z, X, Y)

    X = number of frequency columns
    Y = number of phase-encoding lines
    Z = number of slices
    S = number of shots
    Q = number of diffusion directions
    '''
    return x * phase

def coil_for(x: torch.Tensor, coils: torch.Tensor) -> torch.Tensor:
    '''
    Performs forward coil sensitivity multiplication
    Args:
        x (torch.Tensor): shot split diffusion signal, shape (Q, S, 1, Z, X, Y)
        coils (torch.Tensor): coil sensitivity functions, shape (1, 1, C, Z, X, Y)
    Returns:
        torch.Tensor: shot and coil split diffusion signal, shape  (Q, S, C, Z, X, Y)

    X = number of frequency columns
    Y = number of phase-encoding lines
    Z = number of slices
    C = number of acquisition coils
    S = number of shots
    Q = number of diffusion directions
    '''
    return  x * coils

def fft2c_torch(x: torch.Tensor, dim: tuple[int, int]) -> torch.Tensor:
    '''
    Performs a forward fourier transform
    Args:
        x (torch.Tensor): shot and coil split diffusion signal in image space, shape (Q, S, C, Z, X, Y)
        dim (tuple[int, int]): dimensions to perform Fourier transform on
    Returns:
       torch.Tensor: shot and coil split diffusion signal in k-space, shape (Q, S, C, Z, X, Y) 
    
    X = number of frequency columns
    Y = number of phase-encoding lines
    Z = number of slices
    C = number of acquisition coils
    S = number of shots
    Q = number of diffusion directions            
    '''

    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)

def Multiband_for(x:torch.Tensor, multiband_phase:torch.Tensor)->torch.Tensor:
    '''
    multiplies the k-space diffusion signal with the phase of the multiband acquisition, 
    s.t. the phase between the two slices varies and the images can be split afterwards
    References:
        * Breuer FA, Blaimer M, Heidemann RM, Mueller MF,
          Griswold MA, Jakob PM.
          Controlled aliasing in parallel imagin results in
          higher acceleration (CAIPIRINHA) for multi-slice imaging.
          Magn. Reson. Med. 53:684-691 (2005).

    Args:
        x (torch.Tensor): coil and shot split diffusion data in k-space, shape (Q, S, C, Z, X, Y) 
        phase (torch.Tensor): multiband phases, shape (1, 1, 1, Z, X, Y)
    Returns:
        out (torch.Tensor): multiband, coil and shot split diffusion data in k-space, shape (Q, S, C, 1, X, Y)

    X = number of frequency columns
    Y = number of phase-encoding lines
    C = number of acquisition coils
    S = number of shots
    Q = number of diffusion directions
    '''

    return torch.sum(x *multiband_phase, dim=-3, keepdim=True)

def R(data:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    '''
    Apply undersampling mask to the diffusion signal input
    Args:
        data (torch.Tensor): multiband, coil and shot split diffusion data in k-space, shape (Q, S, C, 1, X, Y)
        mask (torch.Tensor): mask to be applied in frequency and phase encoding direction and for each diffusion direction, shape (Q, S, C, 1, X, Y)
        
    Returns:
        torch.Tensor: masked, multiband, coil and shot split diffusion data in k-space, shape (Q, S, C, 1, X, Y)
    
    X = number of frequency columns
    Y = number of phase-encoding lines
    C = number of acquisition coils
    S = number of shots
    Q = number of diffusion directions
    '''
    
    return data * mask

def tv_loss(x:torch.Tensor, N_z:int, N_x:int, N_y:int, N_latent:int, beta:float = 0.5)->float:
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        N_z (int): Size of z-dimension,
        N_x (int): Size of x-dimension,
        N_y (int): Size of y-dimension,
        N_latent (int): Size of latent dimension,
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    Returns:
        float: TV-loss
    '''

    x = torch.reshape(x, (N_z, N_x, N_y, N_latent))
    diff_x = x[:,1:, :, :] - x[:,:-1, :, :]
    diff_y = x[:,:, 1:, :] - x[:,:, :-1, :]

    # Compute the TV norm by summing the L2 norm of the differences
    tv_x = torch.sum(torch.sqrt(diff_x ** 2 + 1e-8))  # Adding epsilon to avoid sqrt(0)
    tv_y = torch.sum(torch.sqrt(diff_y ** 2 + 1e-8))

    # Combine the results from both dimensions and scale by the weight
    tv_loss = abs(tv_x + tv_y)
    return tv_loss

############################
###    Helper methods    ###
############################

def get_shot_phase(Accel_R, kdat_prep, coil2, ishape, MB, device):  #TODO: adjust or delete
    N_diff, N_z, N_y, N_x = ishape
    acs_shape = [N_y // 4, N_x // 4]
    # ksp_acs = sp.resize(kdat_prep.numpy(), oshape=list(kdat_prep.shape[:-2]) + acs_shape)

    yshift = []
    for b in range(MB):
        yshift.append(b / Accel_R)

    # coils_tensor = sp.to_pytorch(coil2)
    # TR = T.Resize(acs_shape)
    # mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
    # mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
    # mps_acs = mps_acs_r + 1j * mps_acs_i

    # sms_phase_acs = sms.get_sms_phase_shift([MB] + acs_shape, MB=MB, yshift=yshift)

    _, R_shot = muse.MuseRecon(kdat_prep.detach().numpy(), coil2,
                                MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.01, max_iter=30,
                                yshift=yshift,
                                device=sp.Device(-1))
    print('R_shot_phase')
    print(R_shot.shape)
    R_shot_phase = []
    for d in range(N_diff):
        dwi_shot = R_shot[d]
        _, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x])
        R_shot_phase.append(dwi_shot_phase)
    # phs_shots = np.swapaxes(phs_shots, 0, -2)       #(x,s,img,q,y)
    # phs_shots = np.swapaxes(phs_shots, 1, -1)       #(x,y,img,q,s)
    R_shot_phase = np.array(R_shot_phase)
    phs_tensor = torch.tensor(R_shot_phase, dtype=torch.complex64).to(device) #shape (x,y,img,q,s)
    phs_tensor.requires_grad = False
    return phs_tensor

def get_coil(slice_mb_idx: list, coil_path: str, device: str, MB: int)->torch.Tensor:
    '''
    gets the coil files for the kspace data to reconstruct

    Args:
        slice_mb_idx (list): list of slice indexes contained in kspace slice
        coil_path (str): path to coil_file
        device (str): device of reconstruction
        MB (int): multiband factor
    Returns:
        torch.Tensor: coil profiles for slices to reconstruct
    '''
    coils = h5py.File(coil_path, 'r')
    coil_torch = torch.tensor(coils['coil'][:], dtype=torch.complex64).to(device).detach() # c,z,x,y
    coil_torch.requires_grad = False
    for i in range(MB):
        coil = torch.unsqueeze(coil_torch[:,slice_mb_idx[i],...], dim=1)
        if i == 0:
            coil_recon = coil
        else:
            coil_recon = torch.cat((coil_recon, coil), dim=1)
    coil_recon = torch.unsqueeze(coil_recon, dim=0)
    coil_recon = torch.unsqueeze(coil_recon, dim=0)
    coils.close()
    return coil_recon
    
def get_sms_phase(MB: int, N_y: int, N_x: int, Accel_R: int, device: str)->torch.Tensor:
    '''
    gets phase for the sms operation

    Args:
        MB (int): multiband factor
        N_y (int): size of y dimension
        N_x (int): size of x dimension
        Accel_R (int): multiband factor
        device (str): device of reconstruction
        
    Returns:
        torch.Tensor: sms phase tensor
    '''

    yshift = []
    pat = Accel_R #undersampling factor?
    for b in range(MB):
        yshift.append(b / pat)
    mb_phase = get_sms_phase_shift_torch([MB, N_y, N_x], MB=MB, yshift=yshift, device=device)     #z,x,y

    mb_phase = torch.unsqueeze(mb_phase, 0)
    mb_phase = torch.unsqueeze(mb_phase, 0)
    mb_phase = torch.unsqueeze(mb_phase, 0)                                          #x,y,z,1,1,1
    return mb_phase

def get_us_mask(kdata: torch.Tensor, device: str)->torch.Tensor:
    '''
    get the undersampling mask

    Args:
        kdata (torch.Tensor): kspace data of reconstruction
        device (str): device of reconstruction
        
    Returns:
        torch.Tensor: undersampling mask
    '''

    mask = torch.tensor(app._estimate_weights(kdata.detach().cpu().numpy(), None, None, -3), dtype=torch.complex64).to(device).detach()
    mask.requires_grad = False
    return mask

def split_shots_torch(kdat: torch.Tensor, phaenc_axis: int=-2, shots: int=2)->torch.Tensor:
    """
    split shots within one diffusion encoding

    Args:
        kdata (torch.Tensor): kspace data of reconstruction
        phaenc_axis (int): axis of phase encoding
        shots (int): number of shots of the acquisition
        
    Returns:
        torch.Tensor: shot splitted diffusion direction of kspace
    """

    # find valid phase-encoding lines
    kdat1 = torch.swapaxes(kdat, phaenc_axis, 0)
    kdat1.requires_grad = False
    kdat2 = torch.reshape(kdat1, (kdat1.shape[0], -1))
    kdat2.requires_grad = False

    kdat3 = torch.sum(kdat2, axis=1)
    kdat3.requires_grad = False
    sampled_phaenc_ind = torch.Tensor.clone(torch.nonzero(kdat3)).ravel()
    sampled_phaenc_ind.requires_grad = False
    sampled_phaenc_len = len(sampled_phaenc_ind)

    out_shape = torch.empty(shots, kdat2.shape[0], kdat2.shape[1], dtype=torch.complex64)
    out_shape.requires_grad = False
    output = torch.zeros_like(out_shape)
    output.requires_grad = False

    for l in range(sampled_phaenc_len):
        s = l % shots

        ind = sampled_phaenc_ind[l]
        output[s, ind, :] = kdat2[ind, :]

    output = torch.reshape(output, [shots] + list(kdat1.shape))
    output = torch.swapaxes(output, 1, phaenc_axis)

    return output

def vae_reg(model: torch.nn.Module, dwiData: torch.Tensor)->tuple[torch.nn.MSELoss, torch.Tensor]:
    '''
    filters the DWI data using VAE and calculates loss
        
    Args:
        model (torch.nn.Module): loaded VAE model
        dwiData (torch.Tensor): diffusion data
    
    Returns:
        torch.nn.MSELoss: Loss between original DWI data and VAE filtered DWI data
        torch.Tensor: VAE filtered DWI data
    '''
    N_diff,_,_,N_z,N_x,N_y = dwiData.shape
    baseline = dwiData[0]

    dwi_scale = torch.where(baseline!=0, torch.true_divide(dwiData, baseline), torch.zeros_like(dwiData))

    dwi_scaled_mag = abs(dwi_scale).float()
    dwi_scaled_phs = torch.angle(dwiData)

    
    inputData = dwi_scaled_mag.reshape(N_diff,N_z*N_x*N_y)
    inputData = inputData.T


    with torch.no_grad():
        filteredData,_,_ = model(inputData)

    filteredData = filteredData.T
    filteredData = filteredData.reshape(N_diff, 1,1, N_z, N_x, N_y)
    
    filteredData = filteredData * baseline * torch.exp(1j*dwi_scaled_phs)
    criterion   = nn.MSELoss(reduction='sum')
    
    return criterion(torch.view_as_real(dwiData), torch.view_as_real(filteredData)), filteredData

def ShotRecon(y, coils, MB=1, acs_shape=[64, 64],
              max_iter=80,
              yshift=None,
              device=sp.cpu_device, verbose=False):
    """
    Multi shot shot reconstruction:
        1. shot-by-shot SENSE recon;
        2. phase estimation from every shot image;

    Args:
        y (array): zero-filled k-space data with shape:
            [Nshot, Ncoil, Nz_collap, Ny, Nx], where
            - Nshot: # of shots per DWI,
            - Ncoil: # of coils,
            - Nz_collap: # of collapsed slices,
            - Ny: # of phase-encoding lines,
            - Nx: # of readout lines.

        coils (array): coil sensitivity maps with shape:
            [Ncoil, Nz, Ny, Nx], where
            - Nz: # of un-collapsed slices.

        MB (int): multi-band factor
            MB = Nz / Nz_collap.

        acs_shape (tuple of ints): shape of the auto-calibration signal (ACS),
            which is used for the shot-by-shot SENSE recon.

    References:
        * Liu C, Moseley ME, Bammer R.
          Simultaneous phase correction and SENSE reconstruction for navigated multi-shot DWI with non-Cartesian k-space sampling.
          Magn Reson Med 2005;54:1412-1422.

        * Chen NK, Guidon A, Chang HC, Song AW.
          A robust multi-shot strategy for high-resolution diffusion weighted MRI enabled by multiplexed sensitivity-encoding (MUSE).
          NeuroImage 2013;72:41-47.
    """
    Ndiff, Nshot, Ncoil, Nz_collap, Ny, Nx = y.shape
    assert(Nshot > 1)  # MUSE is a multi-shot technique

    _Ncoil, Nz, _Ny, _Nx = coils.shape

    assert ((Ncoil == _Ncoil) and (Ny == _Ny) and (Nx == _Nx))
    assert ((Nz_collap == Nz / MB))

    phi = sms.get_sms_phase_shift([MB, Ny, Nx], MB, yshift=yshift)

    if acs_shape is None:

        ksp_acs = y.copy()
        mps_acs = coils.copy()

    else:

        ksp_acs = sp.resize(y, oshape=list(y.shape[:-2]) + list(acs_shape))

        import torchvision.transforms as T

        coils_tensor = sp.to_pytorch(coils)
        TR = T.Resize(acs_shape, antialias=True)
        mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
        mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
        mps_acs = mps_acs_r + 1j * mps_acs_i

    print('**** MUSE - ksp_acs shape ', ksp_acs.shape)
    print('**** MUSE - mps_acs shape ', mps_acs.shape)

    phs_shot = []
    for z in range(Nz_collap):  # loop over collapsed k-space

        slice_idx = sms.get_uncollap_slice_idx(Nz, MB, z)
        mps_acs_slice = mps_acs[:, slice_idx, ...]

        for d in range(Ndiff):

            print('>> muse on slice ' + str(z).zfill(2) + ' diff ' + str(d).zfill(3))


            xp = device.xp
            ksp_acs = sp.to_device(ksp_acs, device=device)
            mps_acs_slice = sp.to_device(mps_acs_slice, device=device)

            y = sp.to_device(y, device=device)
            coils = sp.to_device(coils, device=device)

            # 1. perform shot-by-shot ACS SENSE recon to estimate phase
            img_acs_shots = []
            for s in range(Nshot):

                ksp = ksp_acs[d, s, :, z, :, :]
                ksp = ksp[..., None, :, :]

                A = muse.sms_sense_linop(ksp, mps_acs_slice, yshift)

                img = muse.sms_sense_solve(A, ksp, lamda=5E-5, tol=0,
                                        max_iter=max_iter, verbose=verbose)

                img_acs_shots.append(img)

            img_acs_shots = xp.array(img_acs_shots)

            # 2. phase estimation from shot images
            _, phs_shots = muse._denoising(img_acs_shots,
                                        full_img_shape=[Ny, Nx])
            phs_shot.append(sp.to_device(phs_shots))

    phs_shot = np.array(phs_shot)

    return phs_shot

def denoising_using_ae(dwi_muse: np.array, ishape: tuple, model: torch.nn.Module, N_latent: int, model_type: str, device: str, bvals: np.array)-> tuple[np.array, np.array]:
    '''
    calculates fractional anisotropy and colored fractional anisotropy of reconstructed DWI data and saves it
        
    Args:
        dwi_muse (np.array): reconstructed DWI data to get denoised
        ishape (tuple): expected shape of input data
        model (torch.nn.Module): AE model to denoise data with
        N_latent (int): size of latent space of network
        model_type (str): type of model
        device (str): device to perform reconstruction on
        bvals (np.array): acquisition used b-values
    
    Returns:
        np.array: denoised dwi_data, shape (Q, Z, X, Y)
        np.array: latent images of AE denoising, shape (L, Z, X, Y)

    Q: q-space dimension
    Z: z dimension
    X: x dimension
    Y: y dimension
    '''

    dwiData = np.squeeze(dwi_muse)

    if dwiData.ndim == 3:
        dwiData = dwiData[:, None, :, :]

    assert dwi_muse.shape == ishape

    b0_mask = bvals > 50

    b0 = dwiData[b0_mask, ...]
    b0_avg = np.mean(b0, dim=0)
    high_angle_entries = abs(np.angle(b0[0,...])*180/np.pi) > 50
    b0_combined = b0[0,...]
    b0_combined[high_angle_entries] = b0_avg[high_angle_entries]

    dwi_scale = np.divide(dwiData, b0_combined,
                        out=np.zeros_like(dwiData),
                        where=dwiData!=0)

    dwi_muse_tensor = torch.tensor(dwi_scale, device=device)

    N_diff,N_z,N_x,N_y = dwi_muse_tensor.shape

    dwi_model_tensor = torch.tensor(np.empty((N_z*N_x*N_y, N_diff)), device=device)

    latent_shape = (N_z*N_x*N_y, N_latent)
    latent_tensor = torch.tensor(np.empty(shape=tuple(latent_shape)), device=device)

    with torch.no_grad():
        qsig = abs(dwi_muse_tensor).float()
        
        inputData = qsig.permute(1,2,3,0)
        inputData = inputData.reshape(N_z*N_x*N_y,N_diff).to(device)

        if model_type == 'VAE':
            dwi_model_tensor ,mu,logvar = model(inputData)
            output = model.reparameterize(mu, logvar)
        elif model_type == 'DAE':
            dwi_model_tensor = model(inputData)
            output = model.encode(inputData)

        dwi_model_tensor = dwi_model_tensor.reshape(N_z, N_x, N_y, N_diff)
        dwi_model_tensor = dwi_model_tensor.permute(3,0,1,2)

        latent_tensor = output.reshape(N_z,N_x,N_y, N_latent)
        latent_tensor = latent_tensor.permute(3,0,1,2)

    unscaled_dwi = dwi_model_tensor.detach().cpu().numpy()
    denoised_dwi = unscaled_dwi * b0_combined

    latent = latent_tensor.detach().cpu().numpy()
    return denoised_dwi, latent


def main():

    # load reconstruction config and save important parameters
    
    DIR = os.path.dirname(os.path.realpath(__file__))

    stream = open('config.yaml', 'r')
    config = yaml.load(stream, Loader)

    muse_recon      = config['muse_recon']
    shot_recon      = config['shot_recon']
    LASER           = config['LASER']
    vae_reg_recon   = config['vae_reg_recon']
    vae_denoise     = config['vae_denoise_recon']

    modelPath       = config['modelPath']
    data_dir        = config['data_directory']
    coil_name       = config['coil_file_name']
    data_name       = config['data_file_name']
    diff_enc_name   = config['diff_enc_file_name']
    save_dir        = config['save_directory']

    slice_idx       = config['slice_index']
    slice_inc       = config['slice_increment']
    device          = config['device']
    reg_weight      = float(config['reg_weight'])

    parser = argparse.ArgumentParser(description="Parser to overwrite slice_idx and slice_inc")
    parser.add_argument("--slice_idx", type=int, default=slice_idx, help="Slice_idx to reconstruct")
    parser.add_argument("--slice_inc", type=int, default=slice_inc, help="slice increment if multiple slice recon")
    args = parser.parse_args()
    slice_idx = args.slice_idx
    slice_inc = args.slice_inc

    print('>> Following reconstructions are run (if True):')
    print('>> Muse reconstruction: ',muse_recon)
    print('>> Shot phase reconstruction: ',shot_recon)
    print('>> LASER: ', LASER)
    print('>> VAE regularized reconstruction: ',vae_reg_recon)
    print('>> VAE denoising: ',vae_denoise)

    deviceDec = torch.device(device)
    print('>> LASER devide:', deviceDec)

    if muse_recon or shot_recon:
        device = sp.Device(-1)               # 0  for gpu, -1 for cpu
        # xp = device.xp
        print('>> Muse devide:', device)

    slice_str = '000'
    print('>> file path:' + data_dir + data_name + slice_str+ '.h5')
    f  = h5py.File(data_dir + data_name + slice_str+ '.h5', 'r')
    MB = f['MB'][()]
    N_slices = f['Slices'][()]
    N_segments = f['Segments'][()]
    N_Accel_PE = f['Accel_PE'][()]
    f.close()

    # number of collapsed slices
    N_slices_collap = N_slices // MB

    # %% run reconstruction
    print('>> slice index: ',slice_idx)
    print('>> slice increment:',slice_inc)
    print('>> multi band factor:',MB)

    if slice_idx >= 0:
        slice_loop = range(slice_idx, slice_idx + slice_inc, 1)
    else:
        slice_loop = range(N_slices_collap)

    for s in slice_loop:
        slice_str = str(s).zfill(3)
        f  = h5py.File(data_dir + data_name +slice_str+'.h5', 'r')
        kdat = f['kdat'][:]
        f.close()

        # correct data shape
        kdat = np.squeeze(kdat)  # 4 dim
        kdat = np.swapaxes(kdat, -2, -3)
        N_diff, N_coils, N_y, N_x = kdat.shape

        # split kdat into shots
        N_diff = kdat.shape[-4]
        kdat_prep = []
        for d in range(N_diff):
            k = retro.split_shots(kdat[d, ...], shots=N_segments)
            kdat_prep.append(k)

        kdat_prep = np.array(kdat_prep)
        kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

        print('>> kdat shape: ', kdat_prep.shape)


        slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

        print('>> slice_mb_idx: ', slice_mb_idx)

        # Get coils
        f = h5py.File(data_dir + coil_name + '.h5', 'r')
        coil = f['coil'][:]
        f.close()
        coil2 = coil[:, slice_mb_idx, :, :]
        coil_path = data_dir + coil_name + '.h5'
        coil_tensor = get_coil(slice_mb_idx,coil_path, device=deviceDec, MB=MB)


        kdat_tensor = torch.tensor(kdat_prep, device=deviceDec)      
        t=time()                                
        sms_phase_tensor = get_sms_phase(MB, N_y, N_x, N_Accel_PE-1, deviceDec) 
        print(sms_phase_tensor.shape)
        print('\n\n>> SMS phase recon time', -t+time())                       
        mask = get_us_mask(kdat_tensor, deviceDec)
        
        f = h5py.File(data_dir + diff_enc_name + '.h5', 'r')
        bvals = f['bvals'][:]
        bvecs = f['bvecs'][:]
        f.close()

        # load model config of used AE and prepare it for reconstruction

        stream = open(modelPath + 'config.yaml', 'r')
        modelConfig = yaml.load(stream, Loader)
        modelType = modelConfig['model']
        model_depth = modelConfig['depth']
        N_latent = modelConfig['latent']
        model_activ_fct = modelConfig['activation_fct']
        b0_mask = None
        if modelConfig['mask_usage']:
            b0_mask = bvals > 50
        ae_dict = {'DAE':ae.DAE, 
                   'VAE':ae.VAE}
        
        model = ae_dict[modelType](b0_mask=b0_mask, input_features=N_diff, latent_features=N_latent, depth=model_depth, activ_fct_str=model_activ_fct, device=deviceDec, reco=True).to(deviceDec)
        model.load_state_dict(torch.load(modelPath + 'train_'+modelType+'_Latent' +str(N_latent).zfill(2) + 'final.pt', map_location=torch.device(deviceDec)))
        
        model = model.float()

        for param in model.parameters():
            param.requires_grad = False

        # Calculate yshift of MB acquisition
        yshift = []
        for b in range(MB):
            yshift.append(b / N_Accel_PE-1)

    #
    # Muse recon
    #
        if muse_recon:
            # kdat_prep = kdat_prep.detach().numpy()
            print('>> Muse reconstruction')
            museFile = h5py.File(save_dir + 'MUSE/MuseRecon_slice_' + slice_str + '.h5', 'w')
            ts=time()         

            acs_shape = [N_y // 4, N_x // 4]

            # Here MUSE was adjusted to return the shot phases (phs_shot in function MuseRecon), not R_shot
            dwi_muse, dwi_shot = muse.MuseRecon(kdat_prep, coil2, MB=MB,
                                    acs_shape=acs_shape,
                                    lamda=0.01, max_iter=30,
                                    yshift=yshift,
                                    device=device)

            dwi_muse = sp.to_device(dwi_muse)
            dwi_shot = sp.to_device(dwi_shot)
            print('>> Muse recon time', time()-ts)

            # store output
            museFile.create_dataset('DWI', data=dwi_muse)
            museFile.create_dataset('Shot_phases', data=dwi_shot)
            museFile.close()
        
    #
    # shot phase recon
    #
        if shot_recon:
            print('>> shot_recon')
            shotFile = h5py.File(save_dir + 'shot_phases/PhaseRecon_slice_' + slice_str + '.h5', 'w')
            ts=time()        

            acs_shape = [N_y // 4, N_x // 4]

            # Here MUSE was adjusted to return the shot phases (phs_shot in function MuseRecon), not R_shot
            shot_phase = ShotRecon(kdat_prep, coil2, MB=MB,
                                    acs_shape=acs_shape,
                                    max_iter=30,
                                    yshift=yshift,
                                    device=device) 

            shot_phase = sp.to_device(shot_phase)
            print('Shot recon time', time()-ts)
            # store output
            
            shotFile.create_dataset('Shot_phases', data=shot_phase)
            shotFile.close()

    #
    # LAtent Space dEcoded Reconstruction (LASER)
    #
        if LASER:

            decFile = h5py.File(save_dir + 'LASER/DecRecon_slice_' + slice_str + '.h5', 'w')

            print('>> Shot phase directory: ' + save_dir + 'shot_phases/PhaseRecon_slice_' + slice_str + '.h5')
            shotFile = h5py.File(save_dir + 'shot_phases/PhaseRecon_slice_' + slice_str + '.h5', 'r')
            shot_phase_tensor = torch.tensor(shotFile['Shot_phases'][:],dtype=torch.complex64, device=deviceDec )
            shotFile.close()

            # Reconstruction of b0**
            print('>> Reconstruction of b0**')
            t=time()

            N_b0 = sum(b0_mask==0)
            b0 = torch.zeros(N_b0,1,1,MB,N_y,N_x, dtype=torch.complex64).to(deviceDec)
            b0.requires_grad  = True

            optimizer   = optim.SGD([b0],lr = 1e-1)

            criterion   = nn.MSELoss(reduction='sum')

            iterations  = 90

            for iter in range(iterations):
                optimizer.zero_grad()
                
                x_multi_shot = Multi_shot_for(b0, shot_phase_tensor[b0_mask==0,...])
                x_coil_split = coil_for(x_multi_shot, coil_tensor[0,...])
                x_k_space = fft2c_torch(x_coil_split, dim=(-2,-1))
                x_mb_combine = Multiband_for(x_k_space, multiband_phase=sms_phase_tensor[0,...])
                x_masked = R(x_mb_combine, mask=mask[b0_mask==0,...])

                loss   = criterion(torch.view_as_real(kdat_tensor[b0_mask==0,...]),torch.view_as_real(x_masked)) + 0.001*criterion(torch.view_as_real(b0),torch.view_as_real(torch.zeros_like(b0)))

                loss.backward()
                optimizer.step()

                running_loss = loss.item()

                if iter % 10 == 0:
                    print(f'>> iteration {iter} / {iterations}, current loss: {running_loss}')

            b0 = torch.squeeze(b0)
            decFile.create_dataset('b0', data=np.array(b0.detach().cpu().numpy()))
            b0_avg = b0.clone().detach()
            b0_avg = torch.mean(b0_avg, dim=0)
            decFile.create_dataset('b0_avg', data=np.array(b0_avg.detach().cpu().numpy()))
            high_angle_entries = abs(torch.angle(b0[0,...])*180/torch.pi) > 50
            b0_combined = b0[0,...].clone().detach()
            b0_combined[high_angle_entries] = b0_avg[high_angle_entries]
            decFile.create_dataset('b0_combined', data=np.array(b0_combined.detach().cpu().numpy()))
            b0_combined.permute(2,1,0).detach()
            b0_combined = torch.reshape(b0_combined, (N_x*N_y*MB,1)).detach()
            b0_avg.permute(2,1,0)
            b0_avg = torch.reshape(b0_avg, (N_x*N_y*MB,1))
            b0 = b0[0,...]
            b0.permute(2,1,0).detach()
            b0 = torch.reshape(b0, (N_x*N_y*MB,1))
            print('>> b0 recon time: ', -t + time())            

            t=time()
            # define latent image tensor
            x_1  = torch.zeros(MB*N_x*N_y,N_latent, dtype=torch.float).to(deviceDec)
            x_1.requires_grad  = True
            
            optimizer   = optim.Adam([x_1],lr = 1e-1)

            criterion   = nn.MSELoss(reduction='sum')

            iterations  = 150
            loss_values = []


            for iter in range(iterations):
                optimizer.zero_grad()
                loss = 0.0
                # batching over coil dimension to reduce size of RAM needed on GPU
                for c in range(N_coils):
                    
                    x= Decoder_for(model, N_x, N_y, MB, N_diff, b0_combined, x_1)             
                    x_multi_shot = Multi_shot_for(x, shot_phase_tensor)
                    x_coil_split = coil_for(x_multi_shot, coil_tensor[:,:,c:c+1,:,:,:])
                    x_k_space = fft2c_torch(x_coil_split, dim=(-2,-1))
                    x_mb_combine = Multiband_for(x_k_space, multiband_phase=sms_phase_tensor)
                    x_masked = R(data=x_mb_combine, mask=mask[:,:,c:c+1,:,:,:])
                    loss += criterion(torch.view_as_real(kdat_tensor[:,:,c:c+1,:,:,:]),torch.view_as_real(x_masked)) 

                if reg_weight > 0:
                    loss_of_tv = reg_weight * tv_loss(x_1, MB, N_x, N_y, N_latent)
                    if iter > 1:
                        loss += loss_of_tv
                loss.backward()
                optimizer.step()

                running_loss = loss.item()
                if np.isnan(running_loss):
                    print('>> Loss is nan')
                    break

                if iter % 10 == 0:
                    print(f'>> iteration {iter} / {iterations}, current loss: {running_loss}')
                    loss_values.append(running_loss)

            print('>> Reconstruction time: ', -t + time())
            lat_img = x_1.detach().cpu().numpy()
            lat_img = np.reshape(lat_img, (MB, N_x, N_y, N_latent))
            lat_img = np.transpose(lat_img, (-1,0,1,2))
            decFile.create_dataset('DWI_latent', data=lat_img)
            x= Decoder_for(model, N_x, N_y, MB, N_diff, b0_combined, x_1)
            decFile.create_dataset('DWI', data=np.array(x.detach().cpu().numpy()))                    
            decFile.close()

    #
    # VAE as regularizer
    #
        if vae_reg_recon:
            t=time()
            print('>> VAE as regularizer')
            lamdas = [0.5]
            print(N_diff)
            for lam in lamdas:
                VAEregFile = h5py.File(save_dir + 'regularizer/VAE_reg_lamda_' + str(lam) + 'recon_slice_' + slice_str + '.h5', 'w')
                print(lam)
                x_1  = torch.zeros((N_diff,1,1,MB,N_y,N_x), dtype=torch.complex64).to(deviceDec)
                
                x_1.requires_grad  = True
                optimizer   = optim.SGD([x_1],lr = 1e-1)

                criterion   = nn.MSELoss(reduction='sum')

                iterations  = 100
                loss_values = []


                for iter in range(iterations):
                    optimizer.zero_grad()
                    
                    x_multi_shot = Multi_shot_for(x_1, shot_phase_tensor)
                    x_coil_split = coil_for(x_multi_shot, coil_tensor)
                    x_k_space = fft2c_torch(x_coil_split, dim=(-2,-1))
                    x_mb_combine = Multiband_for(x_k_space, multiband_phase=sms_phase_tensor)
                    x_masked = R(x_mb_combine, mask=mask)
                    filtered_vae, filtered = vae_reg(model, x_1)
                        
                    loss = criterion(torch.view_as_real(kdat_tensor),torch.view_as_real(x_masked)) + 0.001*criterion(torch.view_as_real(x_1),torch.view_as_real(torch.zeros_like(x_1))) + lam * filtered_vae
                    loss.backward()
                    optimizer.step()
                    
                    running_loss = loss.item()

                    if torch.isnan(loss):
                        break

                    if iter % 10 == 0:
                        print(f'iteration {iter} / {iterations}, current loss: {running_loss}')
                        # VAEregFile.create_dataset('vae_filtered_epoch' + str(iter), data =np.array(filtered.detach().cpu().numpy()))
                        loss_values.append(running_loss)

                VAEregFile.create_dataset('DWI', data=np.array(x_1.detach().cpu().numpy()))
                print('>> VAE regularized recon time', -t+time())
            VAEregFile.close()

    #
    # VAE as denoiser
    #
        if vae_denoise:
            VAEdenoiserFile = h5py.File(save_dir + 'denoiser/VAEDenoiserRecon.h5', 'w')
            print('>> VAE as denoiser')
            dwiData = np.squeeze(dwi_muse)

            if dwiData.ndim == 3:
                dwiData = dwiData[:, None, :, :]

            dwi_scale = np.divide(dwiData, dwiData[0, ...],
                                out=np.zeros_like(dwiData),
                                where=dwiData!=0)

            dwi_muse_tensor = torch.tensor(dwi_scale)

            N_diff,N_z,N_x,N_y = dwi_muse_tensor.shape

            dwi_model_tensor = torch.tensor(np.empty((N_z*N_x*N_y, N_diff)))

            latent_shape = (N_z*N_x*N_y, N_latent)
            latent_tensor = torch.tensor(np.empty(shape=tuple(latent_shape)))

            with torch.no_grad():
                qsig = abs(dwi_muse_tensor).float()
                
                inputData = qsig.permute(1,2,3,0)
                inputData = inputData.reshape(N_z*N_x*N_y,N_diff).to(deviceDec)
                dwi_model_tensor ,_,_ = model(inputData)
                dwi_model_tensor = dwi_model_tensor.reshape(N_z, N_x, N_y, N_diff)
                dwi_model_tensor = dwi_model_tensor.permute(3,0,1,2)
                mu, logvar = model.encode(inputData)                    #N_z*N_x*N_y,Latent
                output = model.reparameterize(mu, logvar)
                latent_tensor = output
                latent_tensor = latent_tensor.reshape(N_z,N_x,N_y, N_latent)
                latent_tensor = latent_tensor.permute(3,0,1,2)

            dwi_vae = dwi_model_tensor.detach().cpu().numpy()
            dwi_vae = dwi_vae * dwiData[0, ...]

            latent = latent_tensor.detach().cpu().numpy()

            VAEdenoiserFile.create_dataset('DWI', data=dwi_vae)
            VAEdenoiserFile.create_dataset('DWI_latent', data=latent)
            VAEdenoiserFile.close()


if __name__ == "__main__":
    main()