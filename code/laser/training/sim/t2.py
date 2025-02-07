"""
This module simulates T2-weighted signal.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np

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
             T2=(0.010, 0.200, 100)):

    T2_array = _linspace_to_array(T2)  # s

    sig = np.zeros((len(TE), len(T2_array)), dtype=float)

    for T2_ind in np.arange(T2[2]):
        T2_val = T2_array[T2_ind]
        z = -1. / T2_val

        sig[:, T2_ind] = np.exp(z * TE)

    return sig
