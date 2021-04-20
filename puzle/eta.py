#! /usr/bin/env python
"""
eta.py
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


def return_eta_ulens_arrs():
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

    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_metadata.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    metadata = np.load(fname)
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, \
           observable1_arr, observable2_arr, observable3_arr


def is_observable_candidate_slope_offset(eta, eta_residual,
                                         slope=1, offset=-1):
    cond = eta_residual >= eta * slope + offset
    return cond


def is_observable_frac_slope_offset(eta, eta_residual,
                                    slope=1, offset=-1):
    is_observable_arr = is_observable_candidate_slope_offset(eta, eta_residual,
                                                             slope=slope, offset=offset)
    frac = np.sum(is_observable_arr) / len(is_observable_arr)
    return frac
