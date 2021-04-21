#! /usr/bin/env python
"""
ulens.py
"""

import numpy as np
import glob
from sqlalchemy.sql.expression import func
from puzle.models import CandidateLevel2
from puzle.utils import return_data_dir


def return_level2_eta_arrs(N_samples=500000):
    cands = CandidateLevel2.query.order_by(func.random()).limit(N_samples).all()
    eta_arr = np.array([c.eta_best for c in cands])
    eta_residual_arr = np.array([c.eta_residual_best for c in cands])
    eta_threshold_low_best = [c.eta_threshold_low_best for c in cands]
    return eta_arr, eta_residual_arr, eta_threshold_low_best


def return_ulens_eta_arrs():
    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_stats.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    data = np.load(fname)

    eta_ulens_arr = data['eta']
    eta_residual_ulens_arr = data['eta_residual']
    observable1_arr = data['observable1']
    observable2_arr = data['observable2']
    observable3_arr = data['observable3']

    metadata = return_ulens_metadata()
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, \
           observable1_arr, observable2_arr, observable3_arr


def return_ulens_stats(observableFlag=True, bhFlag=False):
    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_stats.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    data = np.load(fname)

    cond = np.ones(len(data['eta'])).astype(bool)
    if observableFlag:
        cond *= data['observable3']
    if bhFlag:
        cond *= return_cond_BH()

    stats = {}
    for key in data.keys():
        stats[key] = data[key][cond]

    return stats


def return_ulens_metadata(observableFlag=True, bhFlag=False):
    stats = return_ulens_stats(observableFlag=False,
                               bhFlag=False)

    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_metadata.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    data = np.load(fname)

    cond = np.ones(len(data['tE'])).astype(bool)
    if observableFlag:
        cond *= stats['observable3']
    if bhFlag:
        cond *= return_cond_BH()

    metadata = {}
    for key in data.keys():
        metadata[key] = data[key][cond]

    return metadata


def return_cond_BH(tE_min=150, piE_max=0.08):
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
