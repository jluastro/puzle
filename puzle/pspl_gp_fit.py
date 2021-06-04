#! /usr/bin/env python
"""
pspl_gp_fit.py
"""

import os
import numpy as np
import copy
import pickle
import glob
from astropy.table.row import Row
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


def gather_cand_data(cand, num_max_lightcurves=3):
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


def save_cand_fitter_data(cand, num_max_lightcurves=3):
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


def save_cand_fitter_data_by_id(cand_id, num_max_lightcurves=3):
    cand = CandidateLevel3.query.filter(CandidateLevel3.id==str(cand_id)).first()
    save_cand_fitter_data(cand, num_max_lightcurves=num_max_lightcurves)


def save_all_cand_fitter_data(num_max_lightcurves=3):
    # cands_comp = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
    #                                                     CandidateLevel4.id == CandidateLevel3.id). \
    #     filter(CandidateLevel3.t0_best + CandidateLevel3.tE_best < MJD_finish).\
    #     with_entities(CandidateLevel4.id).\
    #     order_by(CandidateLevel4.id).\
    #     all()
    # cand_ids_comp = [c[0] for c in cands_comp]

    cands_ongoing = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
                                                                CandidateLevel4.id == CandidateLevel3.id). \
        filter(CandidateLevel3.t0_best + CandidateLevel3.tE_best >= MJD_finish).\
        with_entities(CandidateLevel4.id).\
        order_by(CandidateLevel4.id).\
        all()
    cand_ids_ongoing = [c[0] for c in cands_ongoing]

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    # my_cand_ids_comp = np.array_split(cand_ids_comp, size)[rank]
    my_cand_ids_ongoing = np.array_split(cand_ids_ongoing, size)[rank]

    # for i, cand_id in enumerate(my_cand_ids_comp):
    #     if i % 100 == 0:
    #         print('%i) Saving completed cand fitter_data %i / %i' % (rank, i, len(my_cand_ids_comp)))
    #     save_cand_fitter_data_by_id(cand_id, num_max_lightcurves=num_max_lightcurves)

    for i, cand_id in enumerate(my_cand_ids_ongoing):
        if i % 100 == 0:
            print('%i) Saving ongoing cand fitter_data %i / %i' % (rank, i, len(my_cand_ids_ongoing)))
        save_cand_fitter_data_by_id(cand_id, num_max_lightcurves=1)


def load_cand_fitter_data(cand_id):
    out_dir = return_cand_dir(cand_id)
    fname = f'{out_dir}/{cand_id}_fitter_data.dct'
    cand_fitter_data = pickle.load(open(fname, 'rb'))
    return cand_fitter_data


def return_fitter_param_names(data, model_class):
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
    return fitter_param_names


def return_all_param_names(data, model_class):
    multi_filt_params = ['b_sff', 'mag_src', 'mag_base', 'add_err', 'mult_err',
                         'gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0', 'gp_rho',
                         'gp_log_omega04_S0', 'gp_log_omega0', 'add_err', 'mult_err']

    n_phot_sets = 0
    for key in data.keys():
        if 't_phot' in key:
            n_phot_sets += 1

    fitter_param_names = return_fitter_param_names(data, model_class)

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
    outroot = outputfiles_basename

    if remake_fits or not os.path.exists(outroot + '.fits'):
        # Load from text file (and make fits file)
        tab = Table.read(outroot + '.txt', format='ascii')

        # Convert to log(likelihood) since Multinest records -2*logLikelihood
        tab['col2'] /= -2.0

        # Rename the parameter columns. This is hard-coded to match the
        # above run() function.
        tab.rename_column('col1', 'weights')
        tab.rename_column('col2', 'logLike')

        for ff in range(len(all_param_names)):
            cc = 3 + ff
            tab.rename_column('col{0:d}'.format(cc), all_param_names[ff])

        tab.write(outroot + '.fits', overwrite=True)
    else:
        # Load much faster from fits file.
        tab = Table.read(outroot + '.fits')

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


def calc_best_fit(tab, smy, all_param_names, s_idx=0, def_best='maxl'):
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
        tab_best = tab[best][all_param_names]

        return tab_best

    # Use MAP solution
    if def_best.lower() == 'map':
        tab_best = {}
        for n in all_param_names:
            if (n != 'weights' and n != 'logLike'):
                tab_best[n] = smy['MAP_' + n][s_idx]

        return tab_best

    # Use mean solution
    if def_best.lower() == 'mean':
        tab_best = {}
        tab_errors = {}

        for n in all_param_names:
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

        for n in all_param_names:
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
        return best_fit

    all_param_names = return_all_param_names(data, PSPL_Phot_Par_GP_Param2_2)
    tab = load_mnest_results(outputfiles_basename, all_param_names)
    smy = load_mnest_summary(outputfiles_basename, all_param_names)

    best_fit_results = calc_best_fit(tab=tab, smy=smy, all_param_names=all_param_names,
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


def separate_modes(outputfiles_basename):
    # But it seems to work for now...
    mode_file = outputfiles_basename + 'post_separate.dat'

    empty_lines = []
    with open(mode_file, 'r') as orig_file:
        for num, line in enumerate(orig_file, start=0):
            if line == ' \n':
                empty_lines.append(num)

    # Error checking
    if len(empty_lines) % 2 != 0:
        print('SOMETHING BAD HAPPENED!')

    idx_range = int(len(empty_lines) / 2)

    orig_tab = np.loadtxt(mode_file)
    for idx in np.arange(idx_range):
        start_idx = empty_lines[idx * 2 + 1] + 1 - 2 * (idx + 1)
        if idx != np.arange(idx_range)[-1]:
            end_idx = empty_lines[idx * 2 + 2] - 2 * (idx + 1)
            np.savetxt(outputfiles_basename + 'mode' + str(idx) + '.dat', orig_tab[start_idx:end_idx])
        else:
            np.savetxt(outputfiles_basename + 'mode' + str(idx) + '.dat', orig_tab[start_idx:])

    return


def load_mnest_modes(outputfiles_basename, all_param_names, remake_fits=False):
    """Load up the separate modes results into an astropy table.
    """
    # Get all the different mode files

    tab_list = []

    modes = glob.glob(outputfiles_basename + 'mode*.dat')
    if len(modes) < 1:
        print('No modes files! Did you run multinest_utils.separate_mode_files yet?')

    for num, mode in enumerate(modes, start=0):
        mode_root = outputfiles_basename + 'mode' + str(num)

        if remake_fits or not os.path.exists(mode_root + '.fits'):
            # Load from text file (and make fits file)
            tab = Table.read(mode_root + '.dat', format='ascii')

            # Convert to log(likelihood) since Multinest records -2*logLikelihood
            tab['col2'] /= -2.0

            # Rename the parameter columns.
            tab.rename_column('col1', 'weights')
            tab.rename_column('col2', 'logLike')

            for ff in range(len(all_param_names)):
                cc = 3 + ff
                tab.rename_column('col{0:d}'.format(cc), all_param_names[ff])

            tab.write(mode_root + '.fits', overwrite=True)
        else:
            tab = Table.read(mode_root + '.fits')

        tab_list.append(tab)

    return tab_list


def get_num_round_data_points(data_dict):
    N_data = 0
    # Loop through photometry data
    for pp in range(len(data_dict['phot_files'])):
        N_phot_pp = len(set(np.round(data_dict['t_phot{0:d}'.format(pp + 1)])))
        N_data += N_phot_pp

    return N_data


def split_param_filter_index1(s):
    """
    Split a parameter name into the <string><number> components
    where <string> is the parameter name and <number> is the filter
    index (1-based). If there is no number at the end for a filter
    index, then return None for the second argument.

    Returns
    ----------
    param_name : str
        The name of the parameter.
    filt_index : int (or None)
        The 1-based filter index.

    """
    param_name = s.rstrip('123456789')
    if len(param_name) == len(s):
        filt_index = None
    else:
        filt_index = int(s[len(param_name):])

    return param_name, filt_index


def generate_params_dict(params, fitter_param_names):
    """
    Take a list, dictionary, or astropy Row of fit parameters
    and extra parameters and convert it into a well-formed dictionary
    that can be fed straight into a model object.

    The output object will only contain parameters specified
    by name in fitter_param_names. Multi-filter photometry
    parameters are treated specially and grouped together into an
    array such as ['mag_src'] = [mag_src1, mag_src2].

    Input
    ----------
    params : list, dict, Row
        Contains values of parameters. Note that if the
        params are in a list, they need to be in the same
        order as fitter_param_names. If the params are in
        a dict or Row, then order is irrelevant.

    fitter_param_names : list
        The names of the parameters that will be
        delivered, in order, in the output.

    Ouptut
    ----------
    params_dict : dict
        Dictionary of the parameter names and values.

    """
    skip_list = ['weights', 'logLike', 'add_err', 'mult_err']
    multi_list = ['mag_src', 'mag_base', 'b_sff']
    multi_dict = ['gp_log_rho', 'gp_log_S0', 'gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    params_dict = {}

    for i, param_name in enumerate(fitter_param_names):
        # Skip some parameters.
        if any([x in param_name for x in skip_list]):
            continue

        if isinstance(params, (dict, Row)):
            key = param_name
        else:
            key = i

        # Check to see if this is a multi-filter parameter. None if not.
        filt_param, filt_idx = split_param_filter_index1(param_name)

        # Handle global parameters (not filter dependent)
        if filt_idx == None:
            params_dict[param_name] = params[key]
        else:
            # Handle filter dependent parameters... 2 cases (list=required vs. dict=optional)

            if filt_param in multi_list:
                # Handle the filter-dependent fit parameters (required params).
                # They need to be grouped as a list for input into a model.
                if filt_param not in params_dict:
                    params_dict[filt_param] = []

                # Add this filter to our list.
                params_dict[filt_param].append(params[key])

            if filt_param in multi_dict:
                # Handle the optional filter-dependent fit parameters (required params).
                # They need to be grouped as a dicionary for input into a model.
                if filt_param not in params_dict:
                    params_dict[filt_param] = {}

                # Add this filter to our dict. Note the switch to 0-based here.
                params_dict[filt_param][filt_idx - 1] = params[key]

    return params_dict


def get_model(data, params):
    raL, decL = data['raL'], data['decL']
    fitter_param_names = return_fitter_param_names(data, PSPL_Phot_Par_GP_Param2_2)
    params_dict = generate_params_dict(params,
                                       fitter_param_names)

    mod = PSPL_Phot_Par_GP_Param2_2(*params_dict.values(),
                                    raL=raL,
                                    decL=decL)
    return mod


def calc_chi2_round(data, best_par):
    model = get_model(data, best_par)

    n_phot_sets = 0
    for key in data.keys():
        if 't_phot' in key:
            n_phot_sets += 1

    chi2 = 0
    for nn in range(n_phot_sets):
        t_phot = data['t_phot' + str(nn + 1)]
        mag = data['mag' + str(nn + 1)]
        mag_err = data['mag_err' + str(nn + 1)]
        hmjd_round, mag_round = average_xy_on_round_x(t_phot, mag)
        _, magerr_round = average_xy_on_round_x(t_phot, mag_err)
        mag_model = model.get_photometry(hmjd_round, filt_idx=nn)
        chi2_nn = np.sum((mag_round - mag_model)**2. / magerr_round**2.)
        chi2 += chi2_nn

    return chi2


def calc_log_likely(data, best_par):
    model = get_model(data, best_par)

    n_phot_sets = 0
    for key in data.keys():
        if 't_phot' in key:
            n_phot_sets += 1

    lnL_phot = 0.0
    for i in range(n_phot_sets):
        t_phot = data['t_phot' + str(i + 1)]
        mag = data['mag' + str(i + 1)]
        mag_err = data['mag_err' + str(i + 1)]
        lnL_phot += model.log_likely_photometry(t_phot, mag, mag_err, i)

    return lnL_phot


def calc_summary_statistics(cand_id, recomputeFlag=False):
    cand_fitter_data = load_cand_fitter_data(cand_id)
    data = cand_fitter_data['data']
    out_dir = cand_fitter_data['out_dir']
    outputfiles_basename = f'{out_dir}/{cand_id}_'

    # look for cached stats
    stats_fname = f'{outputfiles_basename}stats.dct'
    if os.path.exists(stats_fname) and not recomputeFlag:
        stats = pickle.load(open(stats_fname, 'rb'))
        return stats

    # Get the number of modes.
    summ_tab = Table.read(outputfiles_basename + 'summary.txt', format='ascii')
    N_modes = len(summ_tab) - 1

    # Calculate the number of data points we have all together.
    N_round_data = get_num_round_data_points(data)
    fitter_param_names = return_fitter_param_names(data, PSPL_Phot_Par_GP_Param2_2)
    N_params = len(fitter_param_names)
    N_round_dof = N_round_data - N_params

    # First, we want the statistics for the following types of solutions.
    sol_types = ['maxl', 'mean', 'map', 'median']
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_'}

    all_param_names = return_all_param_names(data, PSPL_Phot_Par_GP_Param2_2)
    separate_modes(outputfiles_basename)
    tab_list = load_mnest_modes(outputfiles_basename, all_param_names)
    smy = load_mnest_summary(outputfiles_basename, all_param_names)

    # Make a deepcopy of this table and set everything to zeros.
    # This will contain our final results.
    stats = copy.deepcopy(smy)
    for col in stats.colnames:
        stats[col] = np.nan

    # Loop through the different modes.
    for nn in range(N_modes):

        # Loop through different types of "solutions"
        for sol in sol_types:
            # Loop through the parameters and get the best fit values.
            foo = calc_best_fit(tab_list[nn], smy, all_param_names, s_idx=nn, def_best=sol)
            if sol == 'maxl' or sol == 'map':
                best_par = foo
            else:
                best_par = foo[0]
                best_parerr = foo[1]

            for param in all_param_names:
                if sol_prefix[sol] + param not in stats.colnames:
                    stats[sol_prefix[sol] + param] = 0.0
                stats[sol_prefix[sol] + param][nn] = best_par[param]

            # Add chi^2 to the table.
            chi2_round = calc_chi2_round(data, best_par)
            if sol_prefix[sol] + 'chi2' not in stats.colnames:
                stats[sol_prefix[sol] + 'chi2'] = 0.0
            stats[sol_prefix[sol] + 'chi2'][nn] = chi2_round

            # Add reduced chi^2 to the table.
            rchi2_round = chi2_round / N_round_dof
            if sol_prefix[sol] + 'rchi2' not in stats.colnames:
                stats[sol_prefix[sol] + 'rchi2'] = 0.0
            stats[sol_prefix[sol] + 'rchi2'][nn] = rchi2_round

            # Add log-likelihood to the table.
            logL = calc_log_likely(data, best_par)
            if sol_prefix[sol] + 'logL' not in stats.colnames:
                stats[sol_prefix[sol] + 'logL'] = 0.0
            stats[sol_prefix[sol] + 'logL'][nn] = logL

            # Next figure out the errors.
            # Only need to do this once.
            if sol == 'median':
                sigma_vals = np.array([0.682689, 0.9545, 0.9973])
                credi_ints_lo = (1.0 - sigma_vals) / 2.0
                credi_ints_hi = (1.0 + sigma_vals) / 2.0
                credi_ints_med = np.array([0.5])
                credi_ints = np.concatenate([credi_ints_med, credi_ints_lo, credi_ints_hi])

                sumweights = np.sum(tab_list[nn]['weights'])
                weights = tab_list[nn]['weights'] / sumweights

                for param in all_param_names:
                    # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
                    tmp = weighted_quantile(tab_list[nn][param], credi_ints, sample_weight=weights)
                    ci_med_val = tmp[0]
                    ci_lo = tmp[1:1 + len(sigma_vals)]
                    ci_hi = tmp[1 + len(sigma_vals):]

                    if 'lo68_' + param not in stats.colnames:
                        stats['lo68_' + param] = 0.0
                        stats['hi68_' + param] = 0.0
                    if 'lo95_' + param not in stats.colnames:
                        stats['lo95_' + param] = 0.0
                        stats['hi95_' + param] = 0.0
                    if 'lo99_' + param not in stats.colnames:
                        stats['lo99_' + param] = 0.0
                        stats['hi99_' + param] = 0.0

                    # Add back in the median to get the actual value (not diff on something).
                    stats['lo68_' + param][nn] = ci_lo[0]
                    stats['hi68_' + param][nn] = ci_hi[0]
                    stats['lo95_' + param][nn] = ci_lo[1]
                    stats['hi95_' + param][nn] = ci_hi[1]
                    stats['lo99_' + param][nn] = ci_lo[2]

        # Get the evidence values out of the _stats.dat file.
        if 'logZ' not in stats.colnames:
            stats['logZ'] = 0.0
        stats['logZ'][nn] = smy['logZ'][nn]

    # Add number of degrees of freedom
    stats['N_dof'] = N_round_dof

    # Sort such that the modes are in reverse order of evidence.
    # Increasing logZ (nan's are at the end)
    zdx = np.argsort(stats['logZ'])
    non_nan = np.where(np.isfinite(stats['logZ'][zdx]))[0]
    zdx = zdx[non_nan[::-1]]

    stats = stats[zdx]
    stats_dct = {c: stats[c].data for c in stats.colnames}
    pickle.dump(stats_dct, open(stats_fname, 'wb'))

    return stats


def calc_summary_stats_to_best_fit(stats, def_best='map'):
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_'}

    if def_best not in sol_prefix:
        print(f'def_best ({def_best}) must be in [%s]' % ', '.join(sol_prefix.keys()))
        return

    def_best_prefix = sol_prefix[def_best]

    best_fit = {}
    keys = [k for k in stats.keys() if def_best_prefix in k]
    for key in keys:
        results_key = key.replace(def_best_prefix, '')
        value = stats[key][0]
        best_fit[results_key] = value

        key_lo = key.replace(def_best_prefix, 'lo68_')
        if key_lo not in stats.keys():
            continue
        value_lo = stats[key_lo][0]
        key_hi = key.replace(def_best_prefix, 'hi68_')
        value_hi = stats[key_hi][0]
        results_key_err = f'{results_key}_err'
        err = np.average([value - value_lo,
                          value_hi - value])
        best_fit[results_key_err] = err

    best_fit['piE'] = np.hypot(best_fit['piE_E'], best_fit['piE_N'])
    a = best_fit['piE_E_err'] * best_fit['piE_E'] / best_fit['piE']
    b = best_fit['piE_N_err'] * best_fit['piE_N'] / best_fit['piE']
    best_fit['piE_err'] = np.sqrt(a**2. + b**2.)

    return best_fit


def fetch_pspl_gp_results(def_best='map', recomputeFlag=False):
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_'}

    if def_best not in sol_prefix:
        print(f'def_best ({def_best}) must be in [%s]' % ', '.join(sol_prefix.keys()))
        return

    cands = CandidateLevel4.query.filter(CandidateLevel4.pspl_gp_fit_finished==True).\
        with_entities(CandidateLevel4.id).order_by(CandidateLevel4.id).all()
    cand_ids = np.array([c[0] for c in cands])

    results = defaultdict(list)
    for i, cand_id in enumerate(cand_ids):
        results['cand_id'].append(cand_id)
        cand_fitter_data = load_cand_fitter_data(cand_id)
        num_lightcurves = cand_fitter_data['fitter_params']['num_lightcurves']
        results['num_lightcurves'].append(num_lightcurves)
        if i % 100 == 0:
            print('Fetching best fit (%i / %i)' % (i, len(cand_ids)))
        stats = calc_summary_statistics(cand_id, recomputeFlag=recomputeFlag)
        best_fit = calc_summary_stats_to_best_fit(stats, def_best=def_best)
        for key, val in best_fit.items():
            results[key].append(val)

    for key in results.keys():
        results[key] = np.array(results[key])

    return dict(results)


def fetch_pspl_gp_results_by_best_fit(def_best='median', errFlag=True, recomputeFlag=False):
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
                err = np.average([best_fit[f'{key}_low_err'],
                                  best_fit[f'{key}_high_err']])
                results[f'{key}_err'].append(err)
    for key in results.keys():
        results[key] = np.array(results[key])
    results['piE'] = np.hypot(results['piE_E'], results['piE_N'])
    if errFlag:
        piE_E_squared_err = 2 * (results['piE_E_err'] / results['piE_E']) * results['piE_E']**2.
        piE_N_squared_err = 2 * (results['piE_N_err'] / results['piE_N']) * results['piE_N']**2.
        piE_squared_frac_err = piE_E_squared_err + piE_N_squared_err
        piE_err = 0.5 * piE_squared_frac_err * results['piE']
        results['piE_err'] = piE_err

    return results


def upload_pspl_gp_results_by_cand_id(cand_id, def_best='map', recomputeFlag=False):
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_'}

    if def_best not in sol_prefix:
        print(f'def_best ({def_best}) must be in [%s]' % ', '.join(sol_prefix.keys()))
        return

    cand_fitter_data = load_cand_fitter_data(cand_id)
    phot_files = cand_fitter_data['data']['phot_files']
    source_id_arr = ['_'.join(k.split('_')[:-1]) for k in phot_files]
    color_arr = [k.split('_')[-1] for k in phot_files]

    stats = calc_summary_statistics(cand_id, recomputeFlag=recomputeFlag)
    best_fit = calc_summary_stats_to_best_fit(stats, def_best=def_best)

    update_dct = {'fit_type_pspl_gp': def_best,
                  'source_id_arr_pspl_gp': source_id_arr,
                  'color_arr_pspl_gp': color_arr}
    cand_keys = [k for k in CandidateLevel4.__dict__.keys() if k.endswith('pspl_gp')]
    for cand_key in cand_keys:
        if 'source_id' in cand_key or 'color' in cand_key:
            continue
        if 'arr' in cand_key:
            if 'err' in cand_key:
                best_fit_keys = [k for k in best_fit if cand_key.replace('_err','').replace('_arr_pspl_gp', '')==k[:-5]]
                best_fit_keys = sorted([k for k in best_fit_keys if 'err' in k])
            else:
                best_fit_keys = [k for k in best_fit if cand_key.replace('_arr_pspl_gp', '')==k[:-1]]
                best_fit_keys = sorted([k for k in best_fit_keys if 'err' not in k])
            update_dct[cand_key] = [best_fit[k] for k in best_fit_keys]
        else:
            best_fit_key = cand_key.replace('_pspl_gp', '')
            if best_fit_key in best_fit:
                update_dct[cand_key] = best_fit[best_fit_key]
                continue

            best_fit_key_one = f'{best_fit_key}1'
            if best_fit_key_one in best_fit:
                update_dct[cand_key] = best_fit[best_fit_key_one]
                continue

            best_fit_key_one_err = best_fit_key.replace('_err', '1_err')
            if best_fit_key_one_err in best_fit:
                update_dct[cand_key] = best_fit[best_fit_key_one_err]
                continue

    db.session.query(CandidateLevel4).filter(CandidateLevel4.id==cand_id).update(update_dct)
    db.session.commit()


if __name__ == '__main__':
    save_all_cand_fitter_data()
