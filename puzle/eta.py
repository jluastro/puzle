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
    observable_arr = data['observable']

    fname = f'{data_dir}/ulens_sample_metadata.total.npz'
    metadata = np.load(fname)
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr
