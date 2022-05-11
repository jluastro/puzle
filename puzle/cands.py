#! /usr/bin/env python
"""
cands.py
"""
import pickle

import numpy as np
import scipy.optimize as op
from astropy.stats import sigma_clip
from sqlalchemy.sql.expression import func
import matplotlib.pyplot as plt

from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.stats import calculate_eta, average_xy_on_round_x
from puzle.models import CandidateLevel2, CandidateLevel3, CandidateLevel4, Source
from puzle.utils import return_figures_dir, return_DR5_dir
from puzle import db

#import pdb

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
    #pdb.set_trace()
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


# def return_sources(cand):
#     source_ids = list(set(cand.source_id_arr))
#     sources = []
#     for source in source_ids:
#
#
#
# def fetch_cand_sources_by_id(cand_id):
#     cand = fetch_cand_by_id(cand_id)
#     if cand is None:
#         objs = None
#     else:
#         objs = return_sources(cand)
#     return objs


def return_eta_residual_slope_offset():
    slope = 3.8187919463087248
    offset = -0.07718120805369133
    return slope, offset


def apply_eta_residual_slope_offset_to_query(query):
    slope, offset = return_eta_residual_slope_offset()
    query = query.filter(CandidateLevel2.eta_residual_best >= CandidateLevel2.eta_best * slope + offset)
    return query


def calculate_chi2_model(param_values, param_names, model_class, data):
    params = {}
    for k, v in zip(param_names, param_values):
        params[k] = v
    model = model_class(**params,
                        raL=data['raL'], decL=data['decL'])
    mag_model = model.get_photometry(data['hmjd'], print_warning=False)

    chi2 = np.sum(((data['mag'] - mag_model) / data['magerr']) ** 2)
    if np.isnan(chi2):
        chi2 = np.inf

    return chi2


def fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec, t0_guess=None, tE_guess=None):
    hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
    _, magerr_round = average_xy_on_round_x(hmjd, magerr)

    # Setup parameter initial guess and list of params
    param_names_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                          'b_sff', 'piE_E', 'piE_N']
    if t0_guess is not None:
        t0 = t0_guess
    else:
        t0 = hmjd_round[np.argmin(mag_round)]
    if tE_guess is not None:
        tE = tE_guess
    else:
        tE = 50

    initial_guess = np.array([t0,
                              0.5,
                              tE,
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
    result = op.minimize(calculate_chi2_model, x0=initial_guess,
                         args=(param_names_to_fit, PSPL_Phot_Par_Param1, data),
                         method='Powell')
    if result.success:
        # gather up best results
        best_fit = result.x
        best_params = {'chi_squared_ulens': result.fun}
        for k, v in zip(param_names_to_fit, best_fit):
            if k == 'tE':
                best_params[k] = abs(v)
            else:
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
        best_params['chi_squared_ulens'] = 0
        best_params['eta_residual'] = 0

    return best_params


def fit_cand_id_to_opt(cand_id, uploadFlag=True, plotFlag=False):
    cand2 = CandidateLevel2.query.filter(CandidateLevel2.id==cand_id).first()
    t0 = cand2.t_0_best
    tE = cand2.t_E_best

    obj = fetch_cand_best_obj_by_id(cand_id)
    hmjd = obj.lightcurve.hmjd
    mag = obj.lightcurve.mag
    magerr = obj.lightcurve.magerr
    ra = obj.ra
    dec = obj.dec

    best_params = fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec,
                                        t0_guess=t0, tE_guess=tE)
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


def return_level3_cut_filters():
    filter0 = CandidateLevel3.tE_best != 0
    filter1 = CandidateLevel3.chi_squared_ulens_best / CandidateLevel3.num_days_best <= 4.805450553176206
    filter2 = CandidateLevel3.delta_hmjd_outside_2tE_best >= 2 * CandidateLevel3.tE_best
    filter3 = CandidateLevel3.chi_squared_flat_outside_2tE_best / CandidateLevel3.num_days_outside_2tE_best <= 3.327056268161699
    filter4 = func.sqrt(func.pow(CandidateLevel3.piE_E_best, 2.) +
                        func.pow(CandidateLevel3.piE_N_best, 2.)) <= 1.4482240516735567
    filter5 = CandidateLevel3.t0_best - CandidateLevel3.tE_best >= 58194.0
    # filter6 = CandidateLevel3.tE_best <= 927.7536173683764
    return filter0, filter1, filter2, filter3, filter4, filter5


def apply_level3_cuts_to_query(query):
    filter0, filter1, filter2, filter3, filter4, filter5 = return_level3_cut_filters()
    query = query.filter(filter0, filter1, filter2,
                         filter3, filter4, filter5)
    return query


def print_level3_cuts():
    filter0, filter1, filter2, filter3, filter4, filter5 = return_level3_cut_filters()
    query = CandidateLevel3.query
    print('Filters up to 0', query.filter(filter0).count(), 'cands')
    print('Filters up to 1', query.filter(filter0, filter1).count(), 'cands')
    print('Filters up to 2', query.filter(filter0, filter1, filter2).count(), 'cands')
    print('Filters up to 3', query.filter(filter0, filter1, filter2, filter3).count(), 'cands')
    print('Filters up to 4', query.filter(filter0, filter1, filter2, filter3, filter4).count(), 'cands')
    print('Filters up to 5', query.filter(filter0, filter1, filter2, filter3, filter4, filter5).count(), 'cands')


def return_sigma_peaks(hmjd, mag, t0, tE, sigma_factor=3, tE_factor=2):
    # perform calculating on day averages
    hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
    # Mask out those points that are within the microlensing event
    ulens_mask = hmjd_round > t0 - tE_factor * tE
    ulens_mask *= hmjd_round < t0 + tE_factor * tE
    # Use this mask to generated a masked array for sigma clipping
    # By applying this mask, the 3-sigma will not be calculated using these points
    mag_round_masked = np.ma.array(mag_round, mask=ulens_mask)
    # Perform the sigma clipping
    mag_round_masked = sigma_clip(mag_round_masked, sigma=3, maxiters=5)
    # This masked array is now a mask that includes BOTH the mirolensing event and
    # the 3-sigma outliers that we want removed. This is the "flats" where we
    # want to calculate the mean and sigma
    mean_flat = mag_round_masked.mean()
    std_flat = mag_round_masked.std()
    # We now add up the number of sigma peaks within tE
    sigma_peaks_inside = np.sum(mag_round[ulens_mask] <= mean_flat - sigma_factor * std_flat)
    if type(sigma_peaks_inside) == np.ma.core.MaskedConstant:
        sigma_peaks_inside = 0
    else:
        sigma_peaks_inside = int(sigma_peaks_inside)
    # And the number of peaks outside
    sigma_peaks_outside = np.sum(mag_round[~ulens_mask] <= mean_flat - sigma_factor * std_flat)
    sigma_peaks_outside += np.sum(mag_round[~ulens_mask] >= mean_flat + sigma_factor * std_flat)
    if type(sigma_peaks_outside) == np.ma.core.MaskedConstant:
        sigma_peaks_outside = 0
    else:
        sigma_peaks_outside = int(sigma_peaks_outside)
    return sigma_peaks_inside, sigma_peaks_outside


def _parse_object_int(attr):
    if attr == 'None':
        return None
    else:
        return int(attr)


def csv_line_to_source(line):
    attrs = line.replace('\n', '').split(',')
    source = Source(id=attrs[0],
                    object_id_g=_parse_object_int(attrs[1]),
                    object_id_r=_parse_object_int(attrs[2]),
                    object_id_i=_parse_object_int(attrs[3]),
                    lightcurve_position_g=_parse_object_int(attrs[4]),
                    lightcurve_position_r=_parse_object_int(attrs[5]),
                    lightcurve_position_i=_parse_object_int(attrs[6]),
                    lightcurve_filename=attrs[7],
                    ra=float(attrs[8]),
                    dec=float(attrs[9]),
                    ingest_job_id=int(attrs[10]))
    return source


def load_source(source_id):
    DR5_dir = return_DR5_dir()
    source_job_id = int(source_id.split('_')[0])
    source_job_prefix = str(source_job_id)[:3]
    sources_fname = f'{DR5_dir}/sources_{source_job_prefix}/sources.{source_job_id:06d}.txt'

    sources_map_fname = sources_fname.replace('.txt', '.sources_map')
    sources_map = pickle.load(open(sources_map_fname, 'rb'))

    f_sources = open(sources_fname, 'r')
    f_sources.seek(sources_map[source_id])
    line_source = f_sources.readline()
    source = csv_line_to_source(line_source)
    f_sources.close()
    return source


def return_level4_cut_filters():
    filter0 = CandidateLevel4.t0_pspl_gp != 0
    filter1 = CandidateLevel4.t0_pspl_gp != 0
    filter2 = CandidateLevel4.t0_pspl_gp != 0
    return filter0, filter1, filter2


def apply_level4_cuts_to_query(query):
    filter0, filter1, filter2 = return_level4_cut_filters()
    query = query.filter(filter0, filter1, filter2)
    return query


def print_level4_cuts():
    filter0, filter1, filter2 = return_level4_cut_filters()
    query = CandidateLevel4.query
    print('Filters up to 0', query.filter(filter0).count(), 'cands')
    print('Filters up to 1', query.filter(filter0, filter1).count(), 'cands')
    print('Filters up to 2', query.filter(filter0, filter1, filter2).count(), 'cands')
