#! /usr/bin/env python
"""
eta.py
"""

import numpy as np
from sqlalchemy.sql.expression import func
from puzle.models import Candidate
from puzle.utils import return_data_dir


def return_eta_arrs(N_samples=500000):
    cands = Candidate.query.order_by(func.random()).limit(N_samples).all()
    eta_arr = np.array([c.eta_best for c in cands])
    eta_residual_arr = np.array([c.eta_residual_best for c in cands])
    eta_threshold_low_best = [c.eta_threshold_low_best for c in cands]
    return eta_arr, eta_residual_arr, eta_threshold_low_best


def return_eta_ulens_arrs():
    data_dir = return_data_dir()
    fname = f'{data_dir}/ulens_sample_etas.total.npz'
    data = np.load(fname)
    eta_ulens_arr = data['eta']
    eta_residual_ulens_arr = data['eta_residual']
    observable1_arr = data['observable1']
    observable2_arr = data['observable2']
    observable3_arr = data['observable3']

    fname = f'{data_dir}/ulens_sample_metadata.total.npz'
    metadata = np.load(fname)
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, \
           observable1_arr, observable2_arr, observable3_arr


def is_observable_candidate(eta, eta_residual,
                            eta_thresh=0.8, slope=1):
    cond1 = eta <= eta_thresh
    cond2 = eta_residual >= eta * slope
    return cond1 * cond2


def is_observable_frac(eta, eta_residual,
                       eta_thresh=0.8, slope=1):
    is_observable_arr = is_observable_candidate(eta, eta_residual,
                                                eta_thresh=eta_thresh, slope=slope)
    frac = np.sum(is_observable_arr) / len(is_observable_arr)
    return frac
