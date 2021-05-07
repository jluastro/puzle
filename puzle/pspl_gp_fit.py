#! /usr/bin/env python
"""
pspl_gp_fit.py
"""

import numpy as np
import pickle
import os
from scipy import stats, optimize

from puzle.models import CandidateLevel3, CandidateLevel4
from puzle.cands import load_source
from puzle.stats import average_xy_on_round_x
from puzle.utils import return_data_dir, MJD_finish
from puzle import db


def make_invgamma_gen(t_arr):
    """
    ADD DESCRIPTION
    t_arr = time array
    """
    a, b = compute_invgamma_params(t_arr)

    return stats.invgamma(a, scale=b)


def solve_for_params(params, x_min, x_max):
    lower_mass = 0.01
    upper_mass = 0.99

    # Trial parameters
    alpha, beta = params

    # Equation for the roots defining params which satisfy the constraint
    cdf_l = stats.invgamma.cdf(x_min, alpha, scale=beta) - lower_mass,
    cdf_u = stats.invgamma.cdf(x_max, alpha, scale=beta) - upper_mass,

    return np.array([cdf_l, cdf_u]).reshape((2,))


def compute_invgamma_params(t_arr):
    """
    Based on function of same name from
    Fran Bartolic's ``caustic`` package:
    https://github.com/fbartolic/caustic
    Returns parameters of an inverse gamma distribution s.t.
    1% of total prob. mass is assigned to values of t < t_{min} and
    1% of total prob. masss  to values greater than t_{tmax}.
    t_{min} is defined to be the median spacing between consecutive
    data points in the time series and t_{max} is the total duration
    of the time series.

    Parameters
    ----------
    t_arr : array
        Array of times
    Returns
    -------
    invgamma_a, invgamma_b : float (?)
        The parameters a,b of the inverse gamma function.
    """

    # Compute parameters for the prior on GP hyperparameters
    med_sep = np.median(np.diff(t_arr))
    tot_dur = t_arr[-1] - t_arr[0]
    results = optimize.fsolve(solve_for_params,
                              (0.001, 0.001),
                              (med_sep, tot_dur))
    invgamma_a, invgamma_b = results

    return invgamma_a, invgamma_b


def return_cand_dir(cand_id):
    data_dir = return_data_dir()
    out_dir = f'{data_dir}/pspl_par_gp_level4_fits/{cand_id}'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_cand_fitter_data(cand):
    source_dct = {}
    data = {'target': cand.id,
            'raL': cand.ra,
            'decL': cand.dec,
            'phot_data': 'ztf',
            'phot_files': [],
            'ast_data': None,
            'ast_files': None}
    fitter_params = {'t0': cand.t0_best,
                     'tE': cand.tE_best}
    num_lightcurves = 0
    for i, (source_id, color) in enumerate(zip(cand.source_id_arr, cand.color_arr)):
        if source_id in source_dct:
            source = source_dct[source_id]
        else:
            source = load_source(source_id)
            source_dct[source_id] = source

        obj = getattr(source.zort_source, f'object_{color}')
        hmjd = obj.lightcurve.hmjd
        n_days = len(set(np.round(hmjd)))
        if n_days < 20:
            continue

        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        _, magerr_round = average_xy_on_round_x(hmjd, magerr)

        idx_data = num_lightcurves + 1
        data[f't_phot{idx_data}'] = hmjd_round
        data[f'mag{idx_data}'] = mag_round
        data[f'mag_err{idx_data}'] = magerr_round
        phot_file = '%s-%s' % (obj.filename, obj.object_id)
        data['phot_files'].append(phot_file)

        flat_cond = hmjd_round < cand.t0_best - 2 * cand.tE_best
        flat_cond += hmjd_round > cand.t0_best + 2 * cand.tE_best
        fitter_params[f'mag_base_{idx_data}'] = np.median(mag_round[flat_cond])
        fitter_params[f'b_sff_{idx_data}'] = cand.b_sff_best

        if i == cand.idx_best:
            fitter_params['idx_data_best'] = idx_data

        num_lightcurves += 1
    fitter_params['num_lightcurves'] = num_lightcurves

    out_dir = return_cand_dir(cand.id)
    cand_fitter_data = {'data': data,
                        'fitter_params': fitter_params,
                        'out_dir': out_dir}
    fname = f'{out_dir}/fitter_data.dct'
    pickle.dump(cand_fitter_data, open(fname, 'wb'))


def save_all_cand_fitter_data():
    cands = db.session.query(CandidateLevel3, CandidateLevel4).\
        filter(CandidateLevel3.id == CandidateLevel4.id,
               CandidateLevel3.t0_best + CandidateLevel3.tE_best < MJD_finish).\
        all()
    for i, (cand3, _) in enumerate(cands):
        if i % 100 == 0:
            print('Saving cand fitter_data %i / %i' % (i, len(cands)))
        save_cand_fitter_data(cand3)


def load_cand_fitter_data(cand_id):
    out_dir = return_cand_dir(cand_id)
    fname = f'{out_dir}/fitter_data.dct'
    cand_fitter_data = pickle.load(open(fname, 'rb'))
    return cand_fitter_data
