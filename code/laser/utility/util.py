"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
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