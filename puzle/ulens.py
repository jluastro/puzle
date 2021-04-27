#! /usr/bin/env python
"""
ulens.py
"""

import numpy as np
import glob
from puzle.utils import return_data_dir, load_stacked_array


def return_ulens_level2_eta_arrs():
    stats = return_ulens_stats(observableFlag=False, bhFlag=False)

    eta_ulens_arr = stats['eta']
    eta_residual_ulens_arr = stats['eta_residual_level2']
    observable1_arr = stats['observable1']
    observable2_arr = stats['observable2']
    observable3_arr = stats['observable3']

    metadata = return_ulens_metadata()
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, \
           observable1_arr, observable2_arr, observable3_arr


def return_ulens_data_fname(prefix):
    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/{prefix}.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    return fname


def return_ulens_data(observableFlag=True, bhFlag=False):
    fname = return_ulens_data_fname('ulens_sample')
    data = load_stacked_array(fname)

    stats = return_ulens_stats(observableFlag=False,
                               bhFlag=False)

    cond = np.ones(len(stats['eta'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
        cond *= stats['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()
    idx_arr = set(np.where(cond==True)[0])

    lightcurve_data = []
    for i, d in enumerate(data):
        if i in idx_arr:
            lightcurve_data.append(d)

    return lightcurve_data


def return_ulens_stats(observableFlag=True, bhFlag=False):
    fname = return_ulens_data_fname('ulens_sample_stats')
    data = np.load(fname)

    cond = np.ones(len(data['eta'])).astype(bool)
    if observableFlag:
        cond *= data['observable3']
        cond *= data['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()

    stats = {}
    for key in data.keys():
        stats[key] = data[key][cond]

    return stats


def return_ulens_metadata(observableFlag=True, bhFlag=False):
    stats = return_ulens_stats(observableFlag=False,
                               bhFlag=False)

    fname = return_ulens_data_fname('ulens_sample_metadata')
    data = np.load(fname)

    cond = np.ones(len(data['tE'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
        cond *= stats['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()

    metadata = {}
    for key in data.keys():
        metadata[key] = data[key][cond]

    return metadata


def return_cond_BH(tE_min=150, piE_max=0.08):
    fname = return_ulens_data_fname('ulens_sample_metadata')
    metadata = np.load(fname)

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'],
                   metadata['piE_N'])
    cond_BH = tE >= tE_min
    cond_BH *= piE <= piE_max
    return cond_BH
