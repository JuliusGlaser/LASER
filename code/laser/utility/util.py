"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Julius Glaser <julius-glaser@gmx.de>
"""

import torch

from torch import Tensor

from typing import Tuple

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os

def _normalize_dims(dims, ndim):
    if dims is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(dims))


def rss(x: Tensor,
        dim: Tuple[int] = (0, ),
        keepdim: bool = False) -> Tensor:

    return torch.sqrt(torch.sum(abs(x)**2, dim=dim, keepdim=keepdim))


def _fftc(x: Tensor,
          dim: Tuple[int] = None,
          norm: str = 'ortho') -> Tensor:

    ndim = x.ndim
    dim = _normalize_dims(dim, ndim)

    tmp = torch.fft.ifftshift(x, dim=dim)
    tmp = torch.fft.fftn(tmp, dim=dim, norm=norm)
    y = torch.fft.fftshift(tmp, dim=dim)

    return y


def _ifftc(x: Tensor,
          dim: Tuple[int] = None,
          norm: str = 'ortho') -> Tensor:

    ndim = x.ndim
    dim = _normalize_dims(dim, ndim)

    tmp = torch.fft.ifftshift(x, dim=dim)
    tmp = torch.fft.ifftn(tmp, dim=dim, norm=norm)
    y = torch.fft.fftshift(tmp, dim=dim)

    return y


def fft(x: Tensor,
        dim: Tuple[int] = None,
        center: bool = True,
        norm: str = 'ortho') -> Tensor:
    if center:
        y = _fftc(x, dim=dim, norm=norm)

    else:
        y = torch.fft.fftn(x, dim=dim, norm=norm)

    return y


def ifft(x: Tensor,
         dim: Tuple[int] = None,
         center: bool = True,
         norm: str = 'ortho') -> Tensor:
    if center:
        y = _ifftc(x, dim=dim, norm=norm)

    else:
        y = torch.fft.ifftn(x, dim=dim, norm=norm)

    return y

def make_gif(numpy_matrix, dim_to_traverse, vmin, vmax, title, filename, interval=1250):
    assert len(numpy_matrix.shape) == 3
    fig, ax = plt.subplots()
    N_img = numpy_matrix.shape[dim_to_traverse]
    matrix_reshaped =  np.moveaxis(numpy_matrix, dim_to_traverse, 0)

    def update(frame):
        if vmax is None and vmin is None:
            ax.imshow(matrix_reshaped[frame,:,:], cmap='gray')
        if vmax is None and vmin is not None:
            ax.imshow(matrix_reshaped[frame,:,:], cmap='gray', vmin=vmin)
        if vmax is not None and vmin is None:
            ax.imshow(matrix_reshaped[frame,:,:], cmap='gray', vmax=vmax)
        else:
            ax.imshow(matrix_reshaped[frame,:,:], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title('Frame ' + str(frame) + '\n' + title)
        return ax

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(N_img), interval=interval, blit=False, repeat=True)
    # Save animation as GIF
    ani.save(filename, writer='pillow')
    return

def load_fonts(dir='/home/hpc/iwbi/iwbi019h/fonts'):
    font_files = font_manager.findSystemFonts(fontpaths=dir)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
        
def set_figure_style(font='Times New Roman', font_size=12, title_size=16, label_size=14,legend_size=12):
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = font_size  # Default font size
    plt.rcParams['axes.titlesize'] = title_size  # Font size of the axes title
    plt.rcParams['axes.labelsize'] = label_size  # Font size of the x and y labels
    plt.rcParams['legend.fontsize'] = legend_size  # Font size of the legend

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
    
def plot_unit_sphere_samples(points):
    # Extract x, y, and z coordinates for plotting
    x_points, y_points, z_points = zip(*points)

    # Create a 3D plot for the unit sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Plot each point on the unit sphere
    ax.scatter(x_points, y_points, z_points, color='red', s=50, label='Samples')

    # Label axes and set the aspect ratio to be equal
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])

    # Show the plot
    plt.legend()
    plt.show()

def plot_vector(ax, points):
    N_vec = points.shape[0]
    if N_vec == 2:
        v1 = points[0,:]
        v2 = points[1,:]
    elif N_vec == 3:
        v1 = points[0,:]
        v2 = points[1,:]
        v3 = points[2,:]


    x_points, y_points, z_points = zip(*points)
    # Plot the sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Plot each vector from the center (0, 0, 0) to the points on the sphere with different colors
    ax.quiver(0, 0, 0, x_points[0], y_points[0], z_points[0], color='red', length=np.linalg.norm(v1), normalize=True, label='v1')
    ax.quiver(0, 0, 0, x_points[1], y_points[1], z_points[1], color='blue', length=np.linalg.norm(v2), normalize=True, label='v2')
    if N_vec == 3:
        ax.quiver(0, 0, 0, x_points[2], y_points[2], z_points[2], color='green', length=np.linalg.norm(v3), normalize=True, label='v3')

    # Add a legend
    ax.legend()

    # Label axes and set the aspect ratio to be equal
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Remove ticks on the axes
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-1,0,1])
    ax.set_box_aspect([1, 1, 1])

def plot_ellipsoid(princ_evec, second_evec, third_evec, evalue, evals_filt_flipped, l1, l2):
    def ellipsoid(center, radii, rotation, num_points=100):
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], rotation) + center
        return x, y, z
    # load gradient directions and b-weightings of sequence
    v1 = np.reshape(princ_evec[l1], (3,1))
    v2 = np.reshape(second_evec[l1][l2], (3,1))
    v3 = np.reshape(third_evec[l1][l2], (3,1))
    mat = np.hstack((v1,v2,v3))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 60)
    u, v = np.meshgrid(u, v)
    print(f"Shape of u: {u.shape}")
    print(f"Shape of v: {v.shape}")
    
    a = evals_filt_flipped[evalue][0]*300
    b = evals_filt_flipped[evalue][1]*300
    c = evals_filt_flipped[evalue][2]*300
    x, y, z = ellipsoid([0,0,0], [a,b,c], mat)
    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Ellipsoid')
    plt.show()

def add_noise(x_clean, scale, noiseType = 'gaussian'):

    if noiseType== 'gaussian':
        x_noisy = x_clean + np.random.normal(loc = 0,
                                            scale = scale,
                                            size=x_clean.shape)
    elif noiseType== 'rician':
        noise1 =np.random.normal(0, scale, size=x_clean.shape)
        noise2 = np.random.normal(0, scale, size=x_clean.shape)
        x_noisy =  np.sqrt((x_clean + noise1) ** 2 + noise2 ** 2)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy