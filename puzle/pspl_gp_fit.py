#! /usr/bin/env python
"""
pspl_gp_fit.py
"""

import numpy as np
import pickle
import os
from scipy import stats

from puzle.cands import load_source
from puzle.stats import average_xy_on_round_x
from puzle.utils import return_data_dir


def return_cand_dir(cand_id):
    data_dir = return_data_dir()
    out_dir = f'{data_dir}/pspl_par_gp_level4_fits/{cand_id}'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_cand_fitter_data(cand):
    idx_best = cand.idx_best
    source_dct = {}
    data = {'target': cand.id,
            'raL': cand.ra,
            'decL': cand.dec,
            'phot_data': 'ztf',
            'phot_files': [],
            'ast_data': None,
            'ast_files': None}
    phot_priors = {}
    idx_data = 1
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

        data[f't_phot{idx_data}'] = hmjd_round
        data[f'mag{idx_data}'] = mag_round
        data[f'mag_err{idx_data}'] = magerr_round
        phot_file = '%s-%s' % (obj.filename, obj.object_id)
        data['phot_files'].append(phot_file)

        phot_priors[f'mag_base{idx_data}'] = stats.norm(np.median(mag), 1.5)
        phot_priors[f'b_sff{idx_data}'] = stats.norm(cand.b_sff_best, 2)

        if i == idx_best:
            phot_priors['idx_data_best'] = idx_data
        idx_data += 1

    out_dir = return_cand_dir(cand.id)
    cand_fitter_data = {'data': data,
                        'phot_priors': phot_priors,
                        'out_dir': out_dir}
    fname = f'{out_dir}/fitter_data.dct'
    pickle.dump(cand_fitter_data, open(fname, 'wb'))


def load_cand_fitter_data(cand_id):
    out_dir = return_cand_dir(cand_id)
    fname = f'{out_dir}/fitter_data.dct'
    cand_fitter_data = pickle.load(open(fname, 'rb'))
    return cand_fitter_data
