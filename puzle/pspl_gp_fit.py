#! /usr/bin/env python
"""
pspl_gp_fit.py
"""

import numpy as np
import pickle
import os

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
        filter(CandidateLevel3.id == CandidateLevel4.id,
               CandidateLevel3.t0_best + CandidateLevel3.tE_best < MJD_finish).\
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


if __name__ == '__main__':
    save_all_cand_fitter_data()
