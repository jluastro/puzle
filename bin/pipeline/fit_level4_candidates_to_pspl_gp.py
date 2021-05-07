#! /usr/bin/env python
"""
fit_level4_candidates_to_pspl_gp.py
"""

import numpy as np
from datetime import datetime
from microlens.jlu import model_fitter
from microlens.jlu.model import PSPL_Phot_Par_GP_Param2_2
from sqlalchemy.sql.expression import func

from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.utils import MJD_finish, get_logger
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


logger = get_logger(__name__)


def fetch_cand():
    insert_db_id()  # get permission to make a db connection

    db.session.execute('LOCK TABLE candidate_level3, candidate_level4 '
                       'IN ROW EXCLUSIVE MODE;')
    cand3, cand4 = db.session.query(CandidateLevel3, CandidateLevel4). \
        filter(CandidateLevel3.id == CandidateLevel4.id,
               CandidateLevel3.t0_best + CandidateLevel3.tE_best < MJD_finish,
               CandidateLevel4.pspl_gp_fit_started == False).\
        order_by(func.random()).\
        with_for_update().\
        first()

    cand_id = cand4.id
    cand4.pspl_gp_fit_started = True
    cand4.pspl_gp_fit_datetime_started = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection
    return cand_id


def finish_cand(cand_id):
    insert_db_id()  # get permission to make a db connection

    cand = db.session.query(CandidateLevel4).filter(
        CandidateLevel4.id == cand_id).one()
    cand.pspl_gp_fit_finished = True
    cand.pspl_gp_fit_datetime_finished = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def fit_level4_cand_to_pspl_gp(cand_id):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cand_fitter_data = load_cand_fitter_data(cand_id)
    data = cand_fitter_data['data']
    fitter_params = cand_fitter_data['fitter_params']
    out_dir = cand_fitter_data['out_dir']
    outputfiles_basename = f'{out_dir}/fit_'

    fitter = model_fitter.PSPL_Solver(data,
                                      PSPL_Phot_Par_GP_Param2_2,
                                      add_error_on_photometry=False,
                                      multiply_error_on_photometry=True,
                                      use_phot_optional_params=True,
                                      importance_nested_sampling=False,
                                      n_live_points=1000,
                                      evidence_tolerance=0.1,
                                      outputfiles_basename=outputfiles_basename)

    # Adjust the priors to encompass both possible solutions
    fitter.priors['t0'] = model_fitter.make_norm_gen(fitter_params['t0'], 25)
    fitter.priors['u0_amp'] = model_fitter.make_norm_gen(0, 0.3)
    fitter.priors['tE'] = model_fitter.make_norm_gen(fitter_params['tE'], 25)
    fitter.priors['piE_E'] = model_fitter.make_norm_gen(-0.02, 0.12)
    fitter.priors['piE_N'] = model_fitter.make_norm_gen(-0.03, 0.13)

    num_lightcurves = fitter_params['num_lightcurves']
    for idx in range(1, num_lightcurves+1):
        fitter.priors[f'b_sff{idx}'] = model_fitter.make_norm_gen(fitter_params[f'b_sff_{idx}'], 0.2)
        fitter.priors[f'mag_base{idx}'] = model_fitter.make_norm_gen(fitter_params[f'mag_base_{idx}'], 0.2)
        fitter.priors[f'gp_log_sigma{idx}'] = model_fitter.make_norm_gen(0, 5)
        fitter.priors[f'gp_rho{idx}'] = model_fitter.make_invgamma_gen(data[f't_phot{idx}'])
        fitter.priors[f'gp_log_omega04_S0{idx}'] = model_fitter.make_norm_gen(np.median(data[f'mag_err{idx}']) ** 2, 5)
        fitter.priors[f'gp_log_omega0{idx}'] = model_fitter.make_norm_gen(0, 5)

    comm.Barrier()
    fitter.solve()

    if rank == 0:
        fitter.plot_dynesty_style(fit_vals='maxl')

        best = fitter.get_best_fit(def_best='maxl')
        model_params = {'t0': best['t0'],
                        'u0_amp': best['u0_amp'],
                        'tE': best['tE'],
                        'piE_E': best['piE_E'],
                        'piE_N': best['piE_N'],
                        'raL': data['raL'],
                        'decL': data['decL']}
        multi_params = ['b_sff', 'mag_base', 'gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']
        for param in multi_params:
            model_params[param] = [best[f'{param}{i}'] for i in range(1, fitter.n_phot_sets + 1)]
        pspl_out = PSPL_Phot_Par_GP_Param2_2(**model_params)
        fitter.plot_model_and_data(pspl_out)

    comm.Barrier()
    finish_cand(cand_id)


def fit_level4_cands_to_pspl_gp(single_job=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    while True:

        if rank == 0:
            cand_id = fetch_cand()
        else:
            cand_id = None
        cand_id = comm.bcast(cand_id, root=0)
        fit_level4_cand_to_pspl_gp(cand_id)
        comm.Barrier()
        if single_job:
            return


if __name__ == '__main__':
    fit_level4_cands_to_pspl_gp()
