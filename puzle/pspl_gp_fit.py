#! /usr/bin/env python
"""
pspl_gp_fit.py
"""

import os
import numpy as np
import pickle
from collections import defaultdict
from microlens.jlu.model import PSPL_Phot_Par_GP_Param2_2
from astropy.table import Table

from puzle.models import CandidateLevel3, CandidateLevel4
from puzle.cands import load_source
from puzle.stats import average_xy_on_round_x
from puzle.utils import return_data_dir, MJD_finish
from puzle import db


def return_cand_dir(cand_id):
    data_dir = return_data_dir()
    out_dir = f'{data_dir}/pspl_par_gp_level4_fits/{cand_id}'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def gather_cand_data(cand, num_max_lightcurves=4):
    source_dct = {}
    n_days_arr = []
    tmp_data_arr = []
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
        n_days_arr.append(n_days)
        tmp_data_arr.append((obj, source_id, color))
    idx_arr = np.argsort(n_days_arr)[::-1][:num_max_lightcurves]
    cand_data_arr = [tmp_data_arr[idx] for idx in idx_arr]
    return cand_data_arr


def fill_data_and_fitter_params(obj, phot_file, idx_data,
                                data_dct, fitter_params_dct):
    hmjd = obj.lightcurve.hmjd
    mag = obj.lightcurve.mag
    magerr = obj.lightcurve.magerr
    hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
    _, magerr_round = average_xy_on_round_x(hmjd, magerr)

    data_dct[f't_phot{idx_data}'] = hmjd_round
    data_dct[f'mag{idx_data}'] = mag_round
    data_dct[f'mag_err{idx_data}'] = magerr_round
    data_dct['phot_files'].append(phot_file)

    flat_cond = hmjd_round < fitter_params_dct['t0'] - 2 * fitter_params_dct['tE']
    flat_cond += hmjd_round > fitter_params_dct['t0'] + 2 * fitter_params_dct['tE']
    if np.sum(flat_cond) < 20:
        mag_base = np.median(mag_round)
    else:
        mag_base = np.median(mag_round[flat_cond])
    fitter_params_dct[f'mag_base_{idx_data}'] = mag_base
    fitter_params_dct[f'b_sff_{idx_data}'] = fitter_params_dct['b_sff']


def save_cand_fitter_data(cand, num_max_lightcurves=4):
    data_dct = {'target': cand.id,
                'raL': cand.ra,
                'decL': cand.dec,
                'phot_data': 'ztf',
                'phot_files': [],
                'ast_data': None,
                'ast_files': None}
    fitter_params_dct = {'t0': cand.t0_best,
                         'tE': cand.tE_best,
                         'u0_amp': cand.u0_amp_best,
                         'piE_E': cand.piE_E_best,
                         'piE_N': cand.piE_N_best,
                         'b_sff': cand.b_sff_best}

    cand_data_arr = gather_cand_data(cand, num_max_lightcurves=num_max_lightcurves)
    fitter_params_dct['num_lightcurves'] = len(cand_data_arr)

    for idx_data, cand_data in enumerate(cand_data_arr, 1):
        obj, source_id, color = cand_data
        phot_file = '%s_%s' % (source_id, color)
        fill_data_and_fitter_params(obj, phot_file, idx_data, data_dct, fitter_params_dct)

    cand_id = cand.id
    out_dir = return_cand_dir(cand_id)
    cand_fitter_data = {'data': data_dct,
                        'fitter_params': fitter_params_dct,
                        'out_dir': out_dir}
    fname = f'{out_dir}/{cand_id}_fitter_data.dct'
    pickle.dump(cand_fitter_data, open(fname, 'wb'))


def save_cand_fitter_data_by_id(cand_id):
    cand = CandidateLevel3.query.filter(CandidateLevel3.id==str(cand_id)).first()
    save_cand_fitter_data(cand)


def save_all_cand_fitter_data():
    cands = db.session.query(CandidateLevel3, CandidateLevel4).\
        filter(CandidateLevel3.id == CandidateLevel4.id).\
        with_entities(CandidateLevel4.id).\
        order_by(CandidateLevel4.id).\
        all()
    cand_ids = [c[0] for c in cands]

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cand_ids = np.array_split(cand_ids, size)[rank]

    for i, cand_id in enumerate(my_cand_ids):
        if i % 100 == 0:
            print('%i) Saving cand fitter_data %i / %i' % (rank, i, len(my_cand_ids)))
        save_cand_fitter_data_by_id(cand_id)


def load_cand_fitter_data(cand_id):
    out_dir = return_cand_dir(cand_id)
    fname = f'{out_dir}/{cand_id}_fitter_data.dct'
    cand_fitter_data = pickle.load(open(fname, 'rb'))
    return cand_fitter_data


def setup_params(data, model_class):
    multi_filt_params = ['b_sff', 'mag_src', 'mag_base', 'add_err', 'mult_err',
                         'gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0', 'gp_rho',
                         'gp_log_omega04_S0', 'gp_log_omega0', 'add_err', 'mult_err']

    n_phot_sets = 0
    phot_params = []

    for key in data.keys():
        if 't_phot' in key:
            n_phot_sets += 1

            # Photometry parameters
            for phot_name in model_class.phot_param_names:
                phot_params.append(phot_name + str(n_phot_sets))

            # Optional photometric parameters -- not all filters
            for opt_phot_name in model_class.phot_optional_param_names:
                phot_params.append(opt_phot_name + str(n_phot_sets))

    fitter_param_names = model_class.fitter_param_names + phot_params

    additional_param_names = []
    for i, param_name in enumerate(model_class.additional_param_names):
        if param_name in multi_filt_params:
            # Handle multi-filter parameters.
            for nn in range(n_phot_sets):
                additional_param_names += [param_name + str(nn + 1)]
        else:
            additional_param_names += [param_name]

    all_param_names = fitter_param_names + additional_param_names
    return all_param_names


def load_mnest_results(outputfiles_basename, all_param_names, remake_fits=False):
    """Load up the MultiNest results into an astropy table.
    """

    if remake_fits or not os.path.exists(outputfiles_basename + '.fits'):
        # Load from text file (and make fits file)
        tab = Table.read(outputfiles_basename + '.txt', format='ascii')

        # Convert to log(likelihood) since Multinest records -2*logLikelihood
        tab['col2'] /= -2.0

        # Rename the parameter columns. This is hard-coded to match the
        # above run() function.
        tab.rename_column('col1', 'weights')
        tab.rename_column('col2', 'logLike')

        for ff in range(len(all_param_names)):
            cc = 3 + ff
            tab.rename_column('col{0:d}'.format(cc), all_param_names[ff])

        tab.write(outputfiles_basename + '.fits', overwrite=True)
    else:
        # Load much faster from fits file.
        tab = Table.read(outputfiles_basename + '.fits')

    return tab


def load_mnest_summary(outputfiles_basename, all_param_names, remake_fits=False):
    """
    Load up the MultiNest results into an astropy table.
    """
    sum_root = outputfiles_basename + 'summary'

    if remake_fits or not os.path.exists(sum_root + '.fits'):
        # Load from text file (and make fits file)
        tab = Table.read(sum_root + '.txt', format='ascii')

        tab.rename_column('col' + str(len(tab.colnames) - 1), 'logZ')
        tab.rename_column('col' + str(len(tab.colnames)), 'maxlogL')

        for ff in range(len(all_param_names)):
            mean = 0 * len(all_param_names) + 1 + ff
            stdev = 1 * len(all_param_names) + 1 + ff
            maxlike = 2 * len(all_param_names) + 1 + ff
            maxapost = 3 * len(all_param_names) + 1 + ff
            tab.rename_column('col{0:d}'.format(mean),
                              'Mean_' + all_param_names[ff])
            tab.rename_column('col{0:d}'.format(stdev),
                              'StDev_' + all_param_names[ff])
            tab.rename_column('col{0:d}'.format(maxlike),
                              'MaxLike_' + all_param_names[ff])
            tab.rename_column('col{0:d}'.format(maxapost),
                              'MAP_' + all_param_names[ff])

        tab.write(sum_root + '.fits', overwrite=True)
    else:
        # Load from fits file, which is much faster.
        tab = Table.read(sum_root + '.fits')

    return tab


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numplt.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numplt.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numplt.percentile.
    :return: numplt.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def calc_best_fit(tab, smy, params, s_idx=0, def_best='maxl'):
    """
    Returns best-fit parameters, where best-fit can be
    median, maxl, or MAP. Default is maxl.

    If best-fit is median, then also return +/- 1 sigma
    uncertainties.

    If best-fit is MAP, then also need to indicate which row of
    summary table to use. Default is s_idx = 0 (global solution).
    s_idx = 1, 2, ... , n for the n different modes.

    tab = self.load_mnest_results()
    smy = self.load_mnest_summary()
    """

    # Use Maximum Likelihood solution
    if def_best.lower() == 'maxl':
        best = np.argmax(tab['logLike'])
        tab_best = tab[best][params]

        return tab_best

    # Use MAP solution
    if def_best.lower() == 'map':
        tab_best = {}
        for n in params:
            if (n != 'weights' and n != 'logLike'):
                tab_best[n] = smy['MAP_' + n][s_idx]

        return tab_best

    # Use mean solution
    if def_best.lower() == 'mean':
        tab_best = {}
        tab_errors = {}

        for n in params:
            if (n != 'weights' and n != 'logLike'):
                tab_best[n] = np.mean(tab[n])
                tab_errors[n] = np.std(tab[n])

        return tab_best, tab_errors

    # Use median solution
    if def_best.lower() == 'median':
        tab_best = {}
        med_errors = {}
        sumweights = np.sum(tab['weights'])
        weights = tab['weights'] / sumweights

        sig1 = 0.682689
        sig2 = 0.9545
        sig3 = 0.9973
        sig1_lo = (1. - sig1) / 2.
        sig2_lo = (1. - sig2) / 2.
        sig3_lo = (1. - sig3) / 2.
        sig1_hi = 1. - sig1_lo
        sig2_hi = 1. - sig2_lo
        sig3_hi = 1. - sig3_lo

        for n in params:
            # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
            tmp = weighted_quantile(tab[n], [0.5, sig1_lo, sig1_hi],
                                    sample_weight=weights)
            tab_best[n] = tmp[0]

            # Switch from values to errors.
            err_lo = tmp[0] - tmp[1]
            err_hi = tmp[2] - tmp[0]

            med_errors[n] = np.array([err_lo, err_hi])

        return tab_best, med_errors


def get_best_fit(cand_id, def_best='maxl', recomputeFlag=False):
    """
    Returns best-fit parameters, where best-fit can be
    median, maxl, or MAP. Default is maxl.

    If best-fit is median, then also return +/- 1 sigma
    uncertainties.
    """
    cand_fitter_data = load_cand_fitter_data(cand_id)
    data = cand_fitter_data['data']
    out_dir = cand_fitter_data['out_dir']
    outputfiles_basename = f'{out_dir}/{cand_id}_'

    best_fit_fname = f'{out_dir}/{cand_id}.{def_best}_fit.dct'
    if os.path.exists(best_fit_fname) and not recomputeFlag:
        best_fit = pickle.load(open(best_fit_fname, 'rb'))
    else:
        all_param_names = setup_params(data, PSPL_Phot_Par_GP_Param2_2)
        tab = load_mnest_results(outputfiles_basename, all_param_names)
        smy = load_mnest_summary(outputfiles_basename, all_param_names)

        best_fit_results = calc_best_fit(tab=tab, smy=smy, params=all_param_names,
                                         s_idx=0, def_best=def_best)

        if def_best == 'median':
            best_fit = dict(best_fit_results[0])
            for param, errs in best_fit_results[1].items():
                best_fit[f'{param}_low_err'] = errs[0]
                best_fit[f'{param}_high_err'] = errs[1]
        else:
            best_fit = dict(best_fit_results)
        pickle.dump(best_fit, open(best_fit_fname, 'wb'))

    return best_fit


def fetch_pspl_gp_results(def_best='median', errFlag=True, recomputeFlag=False):
    if errFlag and def_best!='median':
        print('def_best must be median if errFlag=True')
        return

    cands = CandidateLevel4.query.filter(CandidateLevel4.pspl_gp_fit_finished==True).\
        with_entities(CandidateLevel4.id).order_by(CandidateLevel4.id).all()
    cand_ids = np.array([c[0] for c in cands])

    keys = ['t0',
            'u0_amp',
            'tE',
            'b_sff1',
            'piE_E',
            'piE_N',
            'mag_base1',
            'gp_log_sigma1',
            'gp_log_omega04_S01',
            'gp_log_omega01',
            'gp_log_rho1',
            'gp_log_S01']
    results = defaultdict(list)
    for i, cand_id in enumerate(cand_ids):
        if i % 100 == 0:
            print('Fetching best fit (%i / %i)' % (i, len(cand_ids)))
        best_fit = get_best_fit(cand_id, def_best=def_best,
                                recomputeFlag=recomputeFlag)
        for key in keys:
            results[key].append(best_fit[key])
            if errFlag:
                err = np.average([best_fit[key] - best_fit[f'{key}_low_err'],
                                  best_fit[f'{key}_high_err'] - best_fit[key]])
                results[f'{key}_err'].append(err)
    results['piE'] = np.hypot(results['piE_E'], results['piE_N'])

    return results


if __name__ == '__main__':
    save_all_cand_fitter_data()
