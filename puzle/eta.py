#! /usr/bin/env python
"""
eta.py
"""

import numpy as np
from sqlalchemy.sql.expression import func
from puzle.models import CandidateLevel2


def return_cands_level2_eta_arrs(N_samples=500000):
    cands = CandidateLevel2.query.order_by(func.random()).limit(N_samples).\
        with_entities(CandidateLevel2.eta_best,
                      CandidateLevel2.eta_residual_best,
                      CandidateLevel2.eta_threshold_low_best).all()
    eta_arr = np.array([c[0] for c in cands])
    eta_residual_arr = np.array([c[1] for c in cands])
    eta_threshold_low_best = [c[2] for c in cands]
    return eta_arr, eta_residual_arr, eta_threshold_low_best


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
