#! /usr/bin/env python
"""
cands.py
"""

import numpy as np
import scipy.optimize as op
from sqlalchemy.sql.expression import func
import matplotlib.pyplot as plt

from microlens.jlu.model import PSPL_Phot_Par_Param1

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
    slope = 3.9696969696969697
    offset = -0.09090909090909088
    return slope, offset


def apply_eta_residual_slope_offset_to_query(query):
    slope, offset = return_eta_residual_slope_offset()
    query = query.filter(CandidateLevel2.eta_residual_best >= CandidateLevel2.eta_best * slope + offset)
    return query


def chi2(theta, params_to_fit, model_class, data):
    params = {}
    for k, v in zip(params_to_fit, theta):
        params[k] = v
    model = model_class(**params,
                        raL=data['raL'], decL=data['decL'])

    mag_model = model.get_photometry(data['hmjd'])

    lnL_term1 = -0.5 * ((data['mag'] - mag_model) / data['magerr']) ** 2
    lnL_term2 = -0.5 * np.log(2.0 * np.pi * data['magerr'] ** 2)
    lnL_phot = np.sum(lnL_term1 + lnL_term2)

    lnL_const_phot = -0.5 * np.log(2.0 * np.pi * data['magerr'] ** 2)
    lnL_const_phot = lnL_const_phot.sum()

    # Calculate chi2.
    chi2 = (lnL_phot - lnL_const_phot) / -0.5
    return chi2


def fit_cand_to_ulens(cand_id, uploadFlag=True, plotFlag=False):
    obj = fetch_cand_best_obj_by_id(cand_id)
    hmjd = obj.lightcurve.hmjd
    mag = obj.lightcurve.mag
    magerr = obj.lightcurve.magerr
    ra = obj.ra
    dec = obj.dec

    # Setup parameter initial guess and list of params
    params_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                     'b_sff', 'piE_E', 'piE_N']
    initial_guess = np.array([hmjd[np.argmin(mag)],
                              0.5,
                              50,
                              np.median(mag),
                              1.0,
                              0.25,
                              0.25])

    # instantiate fitter
    data = {'hmjd': hmjd,
            'mag': mag,
            'magerr': magerr,
            'raL': ra,
            'decL': dec}

    # run the optimizer
    result = op.minimize(chi2, x0=initial_guess,
                         args=(params_to_fit, PSPL_Phot_Par_Param1, data),
                         method='Powell')
    if result.success:
        print('** Fit success **')
    else:
        print('** Fit fail **')

    # gather up best results
    best_fit = result.x
    best_params = {}
    for k, v in zip(params_to_fit, best_fit):
        best_params[k.replace('1', '')] = v
    piE = np.hypot(best_params['piE_E'],
                   best_params['piE_N'])

    if uploadFlag and result.success:
        cand = CandidateLevel3.query.filter(CandidateLevel3.id == cand_id).first()
        for param in params_to_fit:
            setattr(cand, f'{param}_best', best_params[param])
        db.session.commit()
        db.session.close()

    # plot results
    if plotFlag:
        # put together a model of best results for plotting
        model = PSPL_Phot_Par_Param1(**best_params, raL=ra, decL=dec)
        hmjd_model = np.linspace(np.min(hmjd),
                                 np.max(hmjd),
                                 2000)
        mag_model = model.get_photometry(hmjd_model)

        fig, ax = plt.subplots()
        ax.clear()
        ax.set_title('tE %.1f | mag_src %.1f | b_sff %.2f | piE %.3f' % (
            best_params['tE'], best_params['mag_src'], best_params['b_sff'], piE))
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
        fit_cand_to_ulens(cand_id)
