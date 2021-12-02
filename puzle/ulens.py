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


def return_ulens_data(observableFlag=True, bhFlag=False,
                      level3Flag=False, sibsFlag=False):
    fname = return_ulens_data_fname('ulens_sample')
    data = load_stacked_array(fname)

    stats = return_ulens_stats(observableFlag=False,
                               bhFlag=False)

    cond = np.ones(len(stats['eta'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
        cond *= stats['tE_level2'] != 0
        cond *= stats['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()
    if level3Flag:
        cond *= stats['tE_level3'] <= 595
        n_days = np.array([len(d) for d in data])
        cond *= stats['chi_squared_ulens_level3'] / n_days <= 2.221
        cond *= np.hypot(stats['piE_E_level3'], stats['piE_N_level3']) <= 2.877
    idx_arr = set(np.where(cond==True)[0])

    lightcurve_data = []
    for i, d in enumerate(data):
        if i in idx_arr:
            lightcurve_data.append(d)

    if not sibsFlag:
        return lightcurve_data

    fname_sibs = fname.replace('ulens_sample', 'ulens_sample.sibs')
    data_sibs = load_stacked_array(fname_sibs)
    lightcurve_sibs_data = []
    for i, d in enumerate(data_sibs):
        if i in idx_arr:
            lightcurve_sibs_data.append(d)

    return lightcurve_data, lightcurve_sibs_data


def return_ulens_stats(observableFlag=True, bhFlag=False, level3Flag=False,
                       sibsFlag=False):
    fname = return_ulens_data_fname('ulens_sample')
    data = load_stacked_array(fname)

    fname = return_ulens_data_fname('ulens_sample_stats')
    stats = np.load(fname)

    cond = np.ones(len(stats['eta'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
        cond *= stats['tE_level2'] != 0
        cond *= stats['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()
    if level3Flag:
        cond *= stats['tE_level3'] <= 595
        n_days = np.array([len(d) for d in data])
        cond *= stats['chi_squared_ulens_level3'] / n_days <= 2.221
        cond *= np.hypot(stats['piE_E_level3'], stats['piE_N_level3']) <= 2.877

    stats_dct = {}
    for key in stats.keys():
        stats_dct[key] = stats[key][cond]

    if not sibsFlag:
        return stats_dct

    fname_sibs = fname.replace('ulens_sample_stats', 'ulens_sample_stats.sibs')
    stats_sibs = np.load(fname_sibs)
    stats_sibs_dct = {}
    for key in stats_sibs.keys():
        stats_sibs_dct[key] = stats_sibs[key][cond]

    return stats_dct, stats_sibs_dct


def return_ulens_metadata(observableFlag=True, bhFlag=False,
                          level3Flag=False, sibsFlag=False):
    stats = return_ulens_stats(observableFlag=False,
                               bhFlag=False)

    fname_metadata = return_ulens_data_fname('ulens_sample_metadata')
    metadata = np.load(fname_metadata)

    fname_data = return_ulens_data_fname('ulens_sample')
    data = load_stacked_array(fname_data)

    cond = np.ones(len(metadata['tE'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
        cond *= stats['tE_level2'] != 0
        cond *= stats['tE_level3'] != 0
    if bhFlag:
        cond *= return_cond_BH()
    if level3Flag:
        cond *= stats['tE_level3'] <= 595
        n_days = np.array([len(d) for d in data])
        cond *= stats['chi_squared_ulens_level3'] / n_days <= 2.221
        cond *= np.hypot(stats['piE_E_level3'], stats['piE_N_level3']) <= 2.877

    metadata_dct = {}
    for key in metadata.keys():
        metadata_dct[key] = metadata[key][cond]

    if not sibsFlag:
        return metadata_dct

    fname_metadata_sibs = fname_metadata.replace('ulens_sample_metadata', 'ulens_sample_metadata.sibs')
    metadata_sibs = np.load(fname_metadata_sibs)
    metadata_sibs_dct = {}
    for key in metadata_sibs.keys():
        metadata_sibs_dct[key] = metadata_sibs[key][cond]

    return metadata_dct, metadata_sibs_dct


def return_cond_BH(tE_min=150, piE_max=0.08):
    fname = return_ulens_data_fname('ulens_sample_metadata')
    metadata = np.load(fname)

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'],
                   metadata['piE_N'])
    cond_BH = tE >= tE_min
    cond_BH *= piE <= piE_max
    return cond_BH
