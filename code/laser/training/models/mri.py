"""
This module implements MRI models

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.jit as jit
import torch.nn as nn

from typing import List, Union


from deepsubspacemri import util
from deepsubspacemri.models.dims import *


class Sense(nn.Module):
    """
    Generalized sensitivity encoding forward modeling

    Args:

    """
    def __init__(self,
                 coils: torch.Tensor,
                 y: torch.Tensor,
                 xshape: List[int] = None,
                 basis: Union[torch.Tensor, nn.Module] = None,
                 phase_echo: torch.Tensor = None,
                 combine_echo: bool = True,
                 phase_slice: torch.Tensor = None,
                 coord: torch.Tensor = None,
                 weights: torch.Tensor = None):
        super(Sense, self).__init__()

        # k-space data shape in accordance with dims.py
        N_time, N_echo, N_coil, N_z, N_y, N_x = y.shape

        # deal with collapsed y even for SMS
        assert(1 == N_z)

        if phase_slice is not None:
            MB = phase_slice.shape[DIM_Z]
        else:
            MB = 1

        # start to construct image shape
        img_shape = [1] + [MB] + [N_y] + [N_x]

        # basis
        self.basis = basis
        if basis is not None:
            assert(N_time == basis.shape[0])
            x_time = basis.shape[1]

        else:
            x_time = N_time

        # echo
        if combine_echo is True:
            ishape = [x_time] + [1] + img_shape
        else:
            ishape = [x_time] + [N_echo] + img_shape

        if xshape is not None:
            self._check_two_shape(ishape, xshape)

        self.xshape = ishape

        # others
        self.y = y
        self.coils = coils

        self.phase_echo = phase_echo
        self.combine_echo = combine_echo
        self.phase_slice = phase_slice
        self.coord = coord

        if weights is None and coord is None:
            weights = (util.rss(y, dim=(DIM_COIL, ), keepdim=True) > 0).type(y.dtype)

        self.weights = weights


    def forward(self, x):

        assert torch.is_tensor(x)
        img_shape = list(x.shape[1:])

        # subspace modeling
        if jit.isinstance(self.basis, torch.Tensor):
            # linear subspace matrix
            _, N_sub = self.basis.shape
            x1 = self.basis @ x.view(x.shape[0], -1)

            x_proj = x1.view([N_sub] + img_shape)

        elif jit.isinstance(self.basis, nn.Module):
            # deep nonlinear subspace
            x1 = x.view(x.shape[0], -1)
            x2 = torch.zeros(self.y.shape[0], dtype=x1.dtype)
            x2 = x2.view(x2.shape[0], -1)
            for n in range(x1.shape[1]):
                px = x1[:, n]
                x2[:, n] = self.basis.decode(px)

            x_proj = x2.view()

        else:
            x_proj = x

        # phase modeling
        if self.phase_echo is not None:
            x_phase = self.phase_echo * x_proj
        else:
            x_phase = x_proj

        # coil sensitivity maps
        x_coils = self.coils * x_phase

        # FFT
        if self.coord is None:
            x_kspace = util.fft(x_coils, dim=(-2, -1))
        else:
            None # TODO: NUFFT

        # SMS
        if self.phase_slice is not None:
            x_kslice = torch.sum(self.phase_slice * x_kspace, dim=DIM_Z, keepdim=True)
        else:
            x_kslice = x_kspace

        # k-space sampling mask
        y = self.weights * x_kslice

        self._check_two_shape(y.shape, self.y.shape)

        return y

    def loss_function(self, y, x, lamda):

        return torch.sum(abs(y - self.forward(x))**2 \
            # + lamda * abs(x)**1)
            + lamda * abs(torch.roll(x, (1, 1), dims=(-2, -1)) - x))


    def _check_two_shape(self, ref_shape, dst_shape):
        for i1, i2 in zip(ref_shape, dst_shape):
            if (i1 != i2):
                raise ValueError('shape mismatch for ref {ref}, got {dst}'.format(
                    ref=ref_shape, dst=dst_shape))
