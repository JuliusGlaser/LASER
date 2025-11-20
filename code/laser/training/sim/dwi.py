"""
This module simulates diffusion MRI signal.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
    Julius Glaser          <julius-glaser@gmx.de>
"""

import numpy as np
from dipy.sims import voxel
from dipy.core import gradients

def get_B(b, g):
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    return - b * np.array([gx**2, 2*gx*gy, gy**2,
                           2*gx*gz, 2*gy*gz, gz**2]).transpose()

def get_evals(N_samples: int)->np.array:
    """
    Gets the combinations of diffusivities for the three eigenvectors of the diffusion tensor.
    Args: 
        N_samples (int): Number of samples from diffusivity interval
    returns:
        np.array: combinations of diffusivities for the three directions of the diffusion tensor
    """
    l1 = np.linspace(0.0001, 3E-3, N_samples)
    l2 = np.linspace(0.0001, 3E-3, N_samples)
    l3 = np.linspace(0.0001, 3E-3, N_samples)

    e1, e2, e3 = np.meshgrid(l1, l2, l3, indexing='ij')
    evals = np.column_stack([e1.ravel(),e2.ravel(),e3.ravel()])

    return evals

def get_perp_vecs(princ_vec: np.array, second_vector_samples: int)-> tuple[np.array, np.array]:
    """
    get perpendicular eigenvectors corresponding to the principle vector.
    The second vector gets as many samples as given in second vector samples and for every 1st and 2nd vector combination a final third vector is calculated
    The second vector is rotated around the first vector.
    Args: 
        princ_vec (np.array): b-values of sequence, 
        second_vector_samples (int): gradients of sequence
    returns:
        np.array: simulated 2nd eigenvectors
        np.array: simulated 3nd eigenvectors
    """

    # Choose a vector not parallel to `v` (to generate an orthogonal vector)
    if np.allclose(princ_vec, [1, 0, 0]):
        t = np.array([0, 1, 0], dtype=float)
    else:
        t = np.array([1, 0, 0], dtype=float)

    # Compute two orthonormal vectors `u1` and `u2` in the plane perpendicular to `v`
    u1 = np.cross(t, princ_vec)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(princ_vec, u1)
    u2 /= np.linalg.norm(u2)

    # Generate points on the circle
    theta = np.linspace(0, 2 * np.pi, second_vector_samples)
    points = np.array([(np.cos(t) * u1 + np.sin(t) * u2) for t in theta])
    scnd_vecs = points
    
    third_vecs = []
    # Normalize the perpendicular vector to make it a unit vector
    for scnd_vec in points:
        third_vec = np.cross(princ_vec, scnd_vec)
        third_vec = third_vec / np.linalg.norm(third_vec)
        third_vecs.append(third_vec)
    third_vecs = np.array(third_vecs)

    return scnd_vecs, third_vecs

def get_D_linspace(D: np.array)->np.array:
    """
    Gets the discretized interval of [D[0], D[1][ with step size D[2].
    Args: 
        D (np.array): Array with 3 entries
    returns:
        np.array: discretized interval according to D
    """
    return np.linspace(D[0], D[1], D[2])

def model_DTI_old(b, g, b0_threshold,
              Dxx=(0    , 3E-3, 9),
              Dxy=(-1E-3, 1E-3, 9),
              Dyy=(0    , 3E-3, 9),
              Dxz=(-1E-3, 1E-3, 9),
              Dyz=(-1E-3, 1E-3, 9),
              Dzz=(0    , 3E-3, 9)
              ):

    B = get_B(b, g)  #(N_diff,6)

    Dxx_grid = get_D_linspace(Dxx)  # 0
    Dxy_grid = get_D_linspace(Dxy)  # 1
    Dyy_grid = get_D_linspace(Dyy)  # 2
    Dxz_grid = get_D_linspace(Dxz)  # 3
    Dyz_grid = get_D_linspace(Dyz)  # 4
    Dzz_grid = get_D_linspace(Dzz)  # 5

    D = np.meshgrid(Dxx_grid, Dxy_grid, Dyy_grid,
                    Dxz_grid, Dyz_grid, Dzz_grid)
    D = np.array(D).reshape((6, -1))

    # diagonal tensors (Dxx, Dyy, Dzz) should be
    # larger than or equal to
    # off-diagonal tensors (Dxy, Dxz, Dyz).
    D_pick1 = []
    for n in range(D.shape[1]):
        d = D[:, n]
        if (d[0] >= d[1] and d[0] >= d[3] and d[0] >= d[4] and
            d[2] >= d[1] and d[2] >= d[3] and d[2] >= d[4] and
            d[5] >= d[1] and d[5] >= d[3] and d[5] >= d[4]):
            D_pick1.append(d)

    D_pick1 = np.transpose(np.array(D_pick1))  # [6, N_D_atoms]

    y = np.exp(np.matmul(B, D_pick1)) # [N_diff, N_D_atoms]

    # q-space signal larger than 1 (b0) violates MR physics.
    y_pick = []
    D_pick2 = []
    for n in range(y.shape[1]): #N_D_atoms
        q = y[:, n]
        if not np.any(q > 1) and not np.any(q > q[0]):
            y_pick.append(q)
            D_pick2.append(D_pick1[:, n])   

    y_pick = np.transpose(np.array(y_pick))  # [N_diff, N_q_atoms]
    D_pick2 = np.transpose(np.array(D_pick2))  # [6, N_q_atoms]

    return y_pick, D_pick2

def model_DTI(b: np.array, g: np.array, b0_threshold: int, diff_samples: int, N_samples_first_evec: int, N_samples_second_evec: int) -> tuple[np.array, np.array]:
    """
    Simulation of DTI model data.
    Args: 
        b (np.array): b-values of sequence, 
        g (np.array): gradients of sequence, 
        b0_threshold (int): threshold value at which low b-values are set equal to b=0 TODO: implement usage,  
        diff_samples (int): number of samples to be taken from discretized value range for diffusivity
        N_samples_first_evec (int): number of used isotropic samples for first eigenvector
        N_samples_second_evec (int): number of used isotropic sampels for second eigenvector
    returns:
        np.array: simulated signals
        np.array: vectors of simulated signals
    """

    gtab = gradients.gradient_table_from_bvals_bvecs(b, g, atol=3e-2)

    # get corresponding eigenvalues for eigenvectors (diffusivity in this direction)
    evals = get_evals(diff_samples)
    # filter out eigenvalues where the principle eigenvector has not the biggest eigenvalue, results in 220 combinations
    condition = (evals[:,1] <= evals[:,0]) & (evals[:,2] <= evals[:,1])
    evals_filt = evals[condition,:]

    # get eigenvectors for model:
    # principle vector, 
    # 2nd vector perpendicular to first, rotating around first one to generate more samples 
    # 3rd vector perpendicular to first two
    princ_evec = sample_from_unit_sphere(N_samples_first_evec)

    second_evec = []
    third_evec = []

    for i in range(princ_evec.shape[0]):
        v = princ_evec[i,:]

        v2, v3 = get_perp_vecs(v, N_samples_second_evec)
        second_evec.append(v2)
        third_evec.append(v3)

    second_evec = np.array(second_evec)
    third_evec = np.array(third_evec)
    signals = []
    evecs_res = []

    print('>> n evals combinations: ', evals_filt.shape[0])
    print('>> n principle evecs: ', princ_evec.shape[0])
    print('>> n second_evecs: ', second_evec.shape[1])

    # simulate diffusion DTI signals and store them
    for i in range(evals_filt.shape[0]):
        for l in range(princ_evec.shape[0]):
            for l2 in range(second_evec.shape[1]):
                v1 = princ_evec[l]
                v2 = second_evec[l][l2]
                v3 = third_evec[l][l2]
                evec = np.vstack([v1,v2,v3]).T
                evecs_res.append((evec.flatten()))
                signals.append(voxel.single_tensor(gtab, S0=1, evals=evals_filt[i], evecs=evec, snr=None, rng=None))
    
    return np.array(signals), np.array(evecs_res)

def model_BAS(b: np.array, g: np.array, b0_threshold: int, N_samples: int = 10,
              diffusivity: tuple[float, float, float]=(0.0001    , 3E-3, 10)
              ) -> tuple[np.array, np.array]:
    """
    Simulation of DTI model data.
    Args: 
        b (np.array): b-values of sequence, 
        g (np.array): gradients of sequence, 
        b0_threshold (int): threshold value at which low b-values are set equal to b=0 TODO: implement usage,  
        N_samples (int): number of used isotropic training samples
        diffusivity (int): number of samples to be taken from discretized value range for diffusivity
        
    returns:
        np.array: simulated signals
        np.array: vectors of simulated signals
    """

    gtab = gradients.gradient_table_from_bvals_bvecs(b, g, atol=3e-2)
    diffusivity_grid = get_D_linspace(diffusivity)
    fractions = get_fractions(nSteps=5)
    angles_table = sample_from_unit_sphere(N_samples)

    # get all possible combinations of sticks simulated in sample_from_unit_sphere
    # results in N_samples x (N_samples-1) combinations
    sticks_combined = []
    for idx, stick1 in enumerate(angles_table):
        for idx2, stick2 in enumerate(angles_table):
            if idx == idx2:
                pass
            else:
                sticks_combined.append([stick1, stick2])

    y_pick = []
    sticks_pick = []
    print('>> fraction combinations: ', fractions.shape[0])
    print('>> sticks combinations: ', len(sticks_combined))
    print('>> diffusivities: ', diffusivity_grid.shape[0])
    # this implementation is suboptimal and takes long to compute for high N_samples
    for frac_i in fractions:
        for d_i in diffusivity_grid:
            for sticks in sticks_combined:
                signal_i, sticks_i = voxel.sticks_and_ball(gtab, 
                                                    d=d_i, 
                                                    S0=1., 
                                                    angles=sticks, 
                                                    fractions=frac_i, 
                                                    snr=None)
                sticks_i = sticks_i.flatten()
                y_pick.append(signal_i)
                sticks_pick.append(sticks_i)

    return np.array(y_pick), np.array(sticks_pick)

def get_fractions(nSteps: int=3)->np.array:
    """
    Get fraction combinations for a 2 stick BAS simulation.
    Args: 
        nSteps (int): number of steps to get step size for fractions      
    returns:
        np.array: combinations of fractions
    """

    stepsize = int(100/nSteps)
    ball_size = stepsize
    while ball_size <= 100:
        numbers = []
        num=0
        while num <= 100-ball_size:
            numbers.append(num)
            num += stepsize
        fraction_array_a = np.array(numbers)
        fraction_array_b = 100 - fraction_array_a - ball_size
        fraction_array2 = np.array((fraction_array_a, fraction_array_b)).T
        try: 
            fraction_array = np.concatenate((fraction_array, fraction_array2))
        except:
            fraction_array = fraction_array2
        ball_size += stepsize
    return fraction_array[0:nSteps]

def sample_from_unit_sphere(n: int)->np.array:
    """
    sample directions from unit sphere.
    Args: 
        n (int): number of samples taken from unit sphere     
    returns:
        np.array: samples (correspond to principle vectors or sticks for DT or BAS)

    implementation taken from https://shorturl.at/CxfVl
    """

    from numpy import arange, pi, sin, cos, arccos
    if n >= 600000:
        epsilon = 214
    elif n>= 400000:
        epsilon = 75
    elif n>= 11000:
        epsilon = 27
    elif n>= 890:
        epsilon = 10
    elif n>= 177:
        epsilon = 3.33
    elif n>= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33

    goldenRatio = (1 + 5**0.5)/2
    i = arange(0, n)
    theta = 2 *pi * i / goldenRatio
    phi = arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    points = np.column_stack((x, y, z))
    return points
    
def _linspace_to_array(linspace_list):

    val0 = linspace_list[0]
    val1 = linspace_list[1]
    leng = linspace_list[2]

    if leng == 1:
        return np.array([val0])
    else:
        step = (val1 - val0) / (leng - 1)
        return val0 + step * np.arange(leng)

def model_t2(TE,
             T2=(0.001, 0.200, 100)):

    T2_array = _linspace_to_array(T2)

    sig = np.zeros((len(TE), 1, 1, 1, 1, 1, len(T2_array)), dtype=float)

    for T2_ind in np.arange(T2[2]):
        T2_val = T2_array[T2_ind]
        z = (-1. / T2_val + 1j * 0.)

        sig[:, 0, 0, 0, 0, 0, T2_ind] = np.exp(z * TE)

    return sig