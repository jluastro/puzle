#! /usr/bin/env python
"""
calculate_eta_residual_slope_offset.py
"""

import numpy as np
import glob

from puzle.utils import return_data_dir
from puzle.eta import return_level2_eta_arrs, return_eta_ulens_arrs, \
    is_observable_frac_slope_offset

import matplotlib.pyplot as plt


def return_cond_BH(tE_min=120, piE_max=0.1):
    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_metadata.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    metadata = np.load(fname)
    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])
    cond_BH = tE >= tE_min
    cond_BH *= piE <= piE_max
    return cond_BH


def calculate_eta_residual_slope_offset():
    eta_arr, eta_residual_arr, _ = return_level2_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, _, _, _, observable_arr = return_eta_ulens_arrs()

    cond = observable_arr == True
    tE_min = 150
    piE_max = 0.08
    cond_BH = return_cond_BH(tE_min=tE_min, piE_max=piE_max)

    slope_arr = np.linspace(1, 7, 150)
    offset_arr = np.linspace(-0.5, 0.25, 150)
    slope_mesh, offset_mesh = np.meshgrid(slope_arr, offset_arr)
    frac_ulens_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    frac_ulens_BH_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    frac_cands_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    for i, offset in enumerate(offset_arr):
        for j, slope in enumerate(slope_arr):
            frac_ulens_mesh[i, j] = is_observable_frac_slope_offset(eta_ulens_arr[cond],
                                                                    eta_residual_ulens_arr[cond],
                                                                    offset=offset,
                                                                    slope=slope)
            frac_ulens_BH_mesh[i, j] = is_observable_frac_slope_offset(eta_ulens_arr[cond * cond_BH],
                                                                       eta_residual_ulens_arr[cond * cond_BH],
                                                                       offset=offset,
                                                                       slope=slope)
            frac_cands_mesh[i, j] = is_observable_frac_slope_offset(eta_arr,
                                                                    eta_residual_arr,
                                                                    offset=offset,
                                                                    slope=slope)
    logfrac_cands_mesh = np.log10(frac_cands_mesh)

    slope_idx_arr = np.argmin(np.abs(frac_ulens_BH_mesh - 0.95), axis=1)
    offset_idx_arr = np.arange(len(offset_arr))

    logfrac_cands_arr = []
    frac_ulens_BH_arr = []
    for slope_idx, offset_idx in zip(slope_idx_arr, offset_idx_arr):
        logfrac_cands_arr.append(logfrac_cands_mesh[offset_idx, slope_idx])
        frac_ulens_BH_arr.append(frac_ulens_BH_mesh[offset_idx, slope_idx])

    idx = np.argmin(logfrac_cands_arr)
    slope = slope_arr[slope_idx_arr][idx]
    offset = offset_arr[offset_idx_arr][idx]

    fig, ax = plt.subplots()
    ax.plot(logfrac_cands_arr, marker='.')
    ax.set_xlabel('idx')
    ax.set_ylabel('LOG [ fraction candidates passed ]')
    ax.axvline(idx)
    ax.set_title('slope = %.2f | offset = %.2f' % (slope, offset),
                 fontsize=10)

    print('slope: ', slope)
    print('offset: ', offset)

    return slope, offset


if __name__ == '__main__':
    calculate_eta_residual_slope_offset()
