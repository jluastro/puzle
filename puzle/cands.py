#! /usr/bin/env python
"""
cands.py
"""

import numpy as np
import scipy.optimize as op
from sqlalchemy.sql.expression import func
import matplotlib.pyplot as plt

from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.stats import calculate_eta, average_xy_on_round_x
from puzle.models import CandidateLevel2, CandidateLevel3, Source
from puzle.utils import return_figures_dir
from puzle import db


def fetch_cand_by_id(cand_id):
    cands = CandidateLevel2.query.filter(CandidateLevel2.id==cand_id).all()
    if len(cands) == 1:
        cand = cands[0]
    else:
        print('No candidates found.')
        cand = None
    return cand


def fetch_cand_by_radec(ra, dec, radius=2):
    cone_filter = CandidateLevel2.cone_search(ra, dec, radius=radius)
    cands = db.session.query(CandidateLevel2).filter(cone_filter).all()
    if len(cands) == 1:
        cand = cands[0]
    elif len(cands) > 1:
        print('Multiple cands within return_cand_by_radec. Returning closest.')
        ra_arr = [c.ra for c in cands]
        dec_arr = [c.dec for c in cands]
        dist_arr = np.hypot(ra_arr-ra, dec_arr-dec)
        idx = np.argmin(dist_arr)
        cand = cands[idx]
    else:
        print('No candidates found.')
        cand = None
    return cand


def return_best_obj(cand):
    idx = cand.idx_best
    source_id = cand.source_id_arr[idx]
    source = Source.query.filter(Source.id==source_id).first()
    color = cand.color_arr[idx]
    obj = getattr(source.zort_source, f'object_{color}')
    return obj


def fetch_cand_best_obj_by_id(cand_id):
    cand = fetch_cand_by_id(cand_id)
    if cand is None:
        obj = None
    else:
        obj = return_best_obj(cand)
    return obj


def return_eta_residual_slope_offset():
    slope = 3.8187919463087248
    offset = -0.07718120805369133
    return slope, offset


def apply_eta_residual_slope_offset_to_query(query):
    slope, offset = return_eta_residual_slope_offset()
    query = query.filter(CandidateLevel2.eta_residual_best >= CandidateLevel2.eta_best * slope + offset)
    return query


def calculate_chi2(param_values, param_names, model_class, data, add_err=0):
    params = {}
    for k, v in zip(param_names, param_values):
        params[k] = v
    model = model_class(**params,
                        raL=data['raL'], decL=data['decL'])

    mag_model = model.get_photometry(data['hmjd'], print_warning=False)

    chi2 = np.sum(((data['mag'] - mag_model) / (data['magerr'] + add_err)) ** 2)
    return chi2


def fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec):
    hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
    _, magerr_round = average_xy_on_round_x(hmjd, magerr)

    # Setup parameter initial guess and list of params
    param_names_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                          'b_sff', 'piE_E', 'piE_N']
    initial_guess = np.array([hmjd[np.argmin(mag_round)],
                              0.5,
                              50,
                              np.median(mag_round),
                              1.0,
                              0.1,
                              0.1])

    # instantiate fitter
    data = {'hmjd': hmjd_round,
            'mag': mag_round,
            'magerr': magerr_round,
            'raL': ra,
            'decL': dec}

    # run the optimizer
    result = op.minimize(calculate_chi2, x0=initial_guess,
                         args=(param_names_to_fit, PSPL_Phot_Par_Param1, data),
                         method='Powell')
    if result.success:
        # gather up best results
        best_fit = result.x
        best_params = {'chi_squared_delta': result.fun}
        for k, v in zip(param_names_to_fit, best_fit):
            best_params[k] = v

        model_params = {k: v for k, v in best_params.items() if k in param_names_to_fit}
        model = PSPL_Phot_Par_Param1(**model_params, raL=ra, decL=dec)
        mag_round_model = model.get_photometry(hmjd_round)
        mag_residual_arr = mag_round - mag_round_model
        cond = ~np.isnan(mag_residual_arr)
        eta_residual = calculate_eta(mag_residual_arr[cond])
        best_params['eta_residual'] = eta_residual
    else:
        best_params = {k: 0 for k in param_names_to_fit}
        best_params['chi_squared_delta'] = 0
        best_params['eta_residual'] = 0

    return best_params


def fit_cand_id_to_opt(cand_id, uploadFlag=True, plotFlag=False):
    obj = fetch_cand_best_obj_by_id(cand_id)
    hmjd = obj.lightcurve.hmjd
    mag = obj.lightcurve.mag
    magerr = obj.lightcurve.magerr
    ra = obj.ra
    dec = obj.dec

    best_params = fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec)
    params_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                     'b_sff', 'piE_E', 'piE_N']

    if uploadFlag:
        cand = CandidateLevel3.query.filter(CandidateLevel3.id == cand_id).first()
        for param, val in best_params.items():
            attr = f'{param}_best'
            try:
                setattr(cand, attr, val)
            except AttributeError:
                pass

        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        eta = calculate_eta(mag_round)
        cand.eta_best = eta

        db.session.commit()
        db.session.close()

    # plot results
    if plotFlag:
        # put together a model of best results for plotting
        hmjd_model = np.linspace(np.min(hmjd),
                                 np.max(hmjd),
                                 2000)
        model_params = {k: v for k, v in best_params.items() if k in params_to_fit}
        model = PSPL_Phot_Par_Param1(**model_params, raL=ra, decL=dec)
        mag_model = model.get_photometry(hmjd_model)

        fig, ax = plt.subplots()
        ax.clear()
        ax.set_title('tE %.1f | mag_src %.1f | b_sff %.2f | piE %.3f' % (
            best_params['tE'], best_params['mag_src'], best_params['b_sff'], best_params['piE']))
        ax.scatter(hmjd, mag, color='b', label='data', s=2)
        ax.plot(hmjd_model, mag_model, color='g', label='model')
        ax.axvline(best_params['t0'], color='k', alpha=.2)
        ax.axvline(best_params['t0'] + best_params['tE'], color='r', alpha=.2)
        ax.axvline(best_params['t0'] - best_params['tE'], color='r', alpha=.2)
        ax.invert_yaxis()
        ax.set_xlabel('hmjd', fontsize=12)
        ax.set_ylabel('mag', fontsize=12)
        ax.legend()

        fname = '%s/%s_lc.png' % (return_figures_dir(), cand_id)
        fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
        print('-- %s saved' % fname)
        plt.close(fig)


def fit_random_cands_to_ulens():
    N_samples = 50
    query = apply_eta_residual_slope_offset_to_query(CandidateLevel2.query)
    cand_ids = query.order_by(func.random()).\
        with_entities(CandidateLevel2.id).\
        limit(N_samples).all()
    cand_ids = [c[0] for c in cand_ids]
    for cand_id in cand_ids:
        fit_cand_id_to_opt(cand_id)


def return_cands_tE_arrs():
    cands23 = db.session.query(CandidateLevel2, CandidateLevel3). \
        filter(CandidateLevel2.id == CandidateLevel3.id). \
        order_by(func.random()). \
        with_entities(CandidateLevel2.t_E_best, CandidateLevel3.tE_best). \
        all()
    tE_minmax_arr = np.array([c[0] for c in cands23])
    tE_opt_arr = np.array([c[1] for c in cands23])
    return tE_minmax_arr, tE_opt_arr


def return_cands_eta_resdiual_arrs():
    cands23 = db.session.query(CandidateLevel2, CandidateLevel3). \
        filter(CandidateLevel2.id == CandidateLevel3.id). \
        with_entities(CandidateLevel2.eta_residual_best, CandidateLevel3.eta_residual_best). \
        all()
    eta_residual_minmax_arr = np.array([c[0] for c in cands23])
    eta_residual_opt_arr = np.array([c[1] for c in cands23])
    return eta_residual_minmax_arr, eta_residual_opt_arr
