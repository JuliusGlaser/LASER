"""
This module simulates diffusion MRI signal.
TODO: more efficient implementation of BAS and DTI

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
    Julius Glaser          <julius-glaser@gmx.de>
"""

import numpy as np
from laser.training.sim.isotrop_vectors import isotropic_vectors as IV60
from dipy.sims import voxel
from dipy.core import gradients

def sphere2cart(r, theta, phi):
    """ Spherical to Cartesian coordinates

    This is the standard physics convention where `theta` is the
    inclination (polar) angle, and `phi` is the azimuth angle.

    Imagine a sphere with center (0,0,0).  Orient it with the z axis
    running south-north, the y axis running west-east and the x axis
    from posterior to anterior.  `theta` (the inclination angle) is the
    angle to rotate from the z-axis (the zenith) around the y-axis,
    towards the x axis.  Thus the rotation is counter-clockwise from the
    point of view of positive y.  `phi` (azimuth) gives the angle of
    rotation around the z-axis towards the y axis.  The rotation is
    counter-clockwise from the point of view of positive z.

    Equivalently, given a point P on the sphere, with coordinates x, y,
    z, `theta` is the angle between P and the z-axis, and `phi` is
    the angle between the projection of P onto the XY plane, and the X
    axis.

    Geographical nomenclature designates theta as 'co-latitude', and phi
    as 'longitude'

    Parameters
    ----------
    r : array_like
       radius
    theta : array_like
       inclination or polar angle
    phi : array_like
       azimuth angle

    Returns
    -------
    x : array
       x coordinate(s) in Cartesian space
    y : array
       y coordinate(s) in Cartesian space
    z : array
       z coordinate

    Notes
    -----
    See these pages:

    * https://en.wikipedia.org/wiki/Spherical_coordinate_system
    * https://mathworld.wolfram.com/SphericalCoordinates.html

    for excellent discussion of the many different conventions
    possible.  Here we use the physics conventions, used in the
    wikipedia page.

    Derivations of the formulae are simple. Consider a vector x, y, z of
    length r (norm of x, y, z).  The inclination angle (theta) can be
    found from: cos(theta) == z / r -> z == r * cos(theta).  This gives
    the hypotenuse of the projection onto the XY plane, which we will
    call Q. Q == r*sin(theta). Now x / Q == cos(phi) -> x == r *
    sin(theta) * cos(phi) and so on.

    We have deliberately named this function ``sphere2cart`` rather than
    ``sph2cart`` to distinguish it from the Matlab function of that
    name, because the Matlab function uses an unusual convention for the
    angles that we did not want to replicate.  The Matlab function is
    trivial to implement with the formulae given in the Matlab help.

    """
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = r * np.cos(theta)
    return x, y, z


def get_B(b, g):
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    return - b * np.array([gx**2, 2*gx*gy, gy**2,
                           2*gx*gz, 2*gy*gz, gz**2]).transpose()

def get_evals(N_samples):
    l1 = np.linspace(0.0001, 3E-3, N_samples)
    l2 = np.linspace(0.0001, 3E-3, N_samples)
    l3 = np.linspace(0.0001, 3E-3, N_samples)

    e1, e2, e3 = np.meshgrid(l1, l2, l3, indexing='ij')
    evals = np.column_stack([e1.ravel(),e2.ravel(),e3.ravel()])

    return evals

def get_perp_vecs(princ_vec):
    num_points = 10
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
    theta = np.linspace(0, 2 * np.pi, num_points)
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

def get_D_linspace(D):
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

def model_DTI(b: np.array, g: np.array, b0_threshold: int, diff_samples: int, N_samples: int) -> tuple[np.array, np.array]:
    """
    Simulation of DTI model data.
    Args: 
        b (np.array): b-values of sequence, 
        g (np.array): gradients of sequence, 
        b0_threshold (int): threshold value at which low b-values are set equal to b=0 TODO: implement usage,  
        diff_samples (int): number of samples to be taken from discretized value range for diffusivity
        N_samples (int): number of used isotropic training samples TODO: yet to be implemented, for now fixed at 60
    returns:
        np.array: simulated signals
        np.array: vectors of simulated signals
    """

    gtab = gradients.gradient_table_from_bvals_bvecs(b, g, atol=3e-2)
    
    # get corresponding eigenvalues for eigenvectors (diffusivity in this direction)
    evals = get_evals(diff_samples)
    condition = (evals[:,1] <= evals[:,0]) & (evals[:,2] <= evals[:,1])
    evals_filt = evals[condition,:]

    # get eigenvectors for model:
    # principle vector, 
    # 2nd vector perpendicular to first, rotating around first one to generate more samples TODO: make number of samples variable for second vector
    # 3rd vector perpendicular to first two

    princ_evec = np.array(IV60)
    second_evec = []
    third_evec = []

    for i in range(princ_evec.shape[0]):
        v = princ_evec[i,:]

        v2, v3 = get_perp_vecs(v)
        second_evec.append(v2)
        third_evec.append(v3)

    second_evec = np.array(second_evec)
    third_evec = np.array(third_evec)
    signals = []
    evecs_res = []

    # simulate diffusion DTI signals and store them signals
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

def model_BAS(b: np.array, g: np.array, b0_threshold: int, N_samples: int =None,
              diffusivity: tuple[float, float, float]=(0.0001    , 3E-3, 10)
              ) -> tuple[np.array, np.array]:
    """
    Simulation of DTI model data.
    Args: 
        b (np.array): b-values of sequence, 
        g (np.array): gradients of sequence, 
        b0_threshold (int): threshold value at which low b-values are set equal to b=0 TODO: implement usage,  
        N_samples (int): number of used isotropic training samples TODO: yet to be implemented, for now fixed at 60
        diffusivity (int): number of samples to be taken from discretized value range for diffusivity
        
    returns:
        np.array: simulated signals
        np.array: vectors of simulated signals
    """

    gtab = gradients.gradient_table_from_bvals_bvecs(b, g, atol=3e-2)
    diffusivity_grid = get_D_linspace(diffusivity)
    fractions = get_fractions(nSteps=5)

    # get the whole upper half sphere as possibilites for first stick
    # angles_table = sample_from_unit_sphere(N_samples)
    angles_table = IV60

    sticks_combined = []
    for stick1 in angles_table:
        for stick2 in angles_table:
            if stick1 == stick2:
                pass
            else:
                sticks_combined.append([stick1, stick2])
                if N_samples is not None:
                    if len(sticks_combined)> N_samples:
                        break

    y_pick = []
    sticks_pick = []
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
    size = len(y_pick)
    # #create zero signals for noise
    # for i in range(int(size*0.01)):
    #     signal_i = np.zeros_like(y_pick[0])
    #     y_pick.append(signal_i)
    #     sticks_pick.append([0,0,0,0,0,0])

    return np.array(y_pick), np.array(sticks_pick)

def get_fractions(nSteps: int=3)->np.array:
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
    return fraction_array

def sample_from_unit_sphere(num_pts):
    #TODO: implement correctly
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    combined = []
    for i in range(len(phi)):
        combined.append(((theta[i]*180/np.pi)%360, phi[i]*180/np.pi))

    return combined
    

def calc_BAS_res(b, g, D1, D2, d):
    B = get_B(b, g)
    f0 = 0
    f1 = f2 = 0.5
    # S = f0*np.exp(-b*d)
    S = f1*np.exp(np.matmul(B,D1)*d)
    S += f2*np.exp(np.matmul(B,D2)*d)

    return S

def calc_DTI_res(b,g,D1,d):
    if b.ndim == 1:
        b = b[:, np.newaxis]
    B = get_B(b, g)
    S = np.exp(np.matmul(B,D1)*d)

    return S

def convert6To9(Field):
    assert Field.shape[0] == 6
    if len(Field.shape) == 3:
        rows = Field.shape[1]
        cols = Field.shape[2]
        outField = np.zeros((3,3,rows,cols))
        for row in range(rows):
            for col in range(cols):
                currentD = Field[:,row,col]
                Din9 = np.array([[currentD[0], currentD[1], currentD[3]],
                                [currentD[1], currentD[2], currentD[4]],
                                [currentD[3], currentD[4], currentD[5]]])
                outField[:,:,row,col] = Din9
    else:
        outField = np.zeros((3,3))
        currentD = Field
        Din9 = np.array([[currentD[0], currentD[1], currentD[3]],
                        [currentD[1], currentD[2], currentD[4]],
                        [currentD[3], currentD[4], currentD[5]]])
        outField[:,:] = Din9
    return outField

def convert9To6(Field):
    assert Field.shape[0] == 3
    assert Field.shape[1] == 3
    if len(Field.shape) == 4:
        rows = Field.shape[2]
        cols = Field.shape[3]
        outField = np.zeros((6,rows,cols))
        for row in range(rows):
            for col in range(cols):
                currentD = Field[:,row,col]
                Din6 = np.array([currentD[0][0], currentD[1][0], currentD[1][1], currentD[2][0], currentD[1][2], currentD[2][2]])
                outField[:,:,row,col] = Din6
    else:
        outField = np.zeros((6))
        currentD = Field
        Din6 = np.array([currentD[0][0], currentD[1][0], currentD[1][1], currentD[2][0], currentD[1][2], currentD[2][2]])
        outField = Din6
    return outField

def rotx(angle):
    angle = angle/180*np.pi
    rotMatrix = np.array([[1, 0,0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
    return rotMatrix

def roty(angle):
    angle = angle/180*np.pi
    rotMatrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
    return rotMatrix

def rotz(angle):
    angle = angle/180*np.pi
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
    return rotMatrix

def rotTensorAroundX(D, angle):
    D = convert6To9(D)
    rotatedD = np.matmul(rotx(angle),D)
    rotatedD = np.matmul(rotatedD,rotx(angle).T)
    rotatedD = convert9To6(rotatedD)
    return rotatedD

def rotTensorAroundY(D, angle):
    D = convert6To9(D)
    rotatedD = np.matmul(roty(angle),D)
    rotatedD = np.matmul(rotatedD,roty(angle).T)
    rotatedD = convert9To6(rotatedD)
    return rotatedD

def rotTensorAroundZ(D, angle):
    D = convert6To9(D)
    rotatedD = np.matmul(rotz(angle),D)
    rotatedD = np.matmul(rotatedD,rotz(angle).T)
    rotatedD = convert9To6(rotatedD)
    return rotatedD
