#! /usr/bin/env python
"""
fit_level4_candidates_to_pspl_gp.py
"""

import sys
import numpy as np
from datetime import datetime
from microlens.jlu import model_fitter
from microlens.jlu.model import PSPL_Phot_Par_GP_Param2_2
from sqlalchemy.sql.expression import func
import logging

from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.utils import MJD_finish, get_logger
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


logger = get_logger(__name__)


def fetch_cand(slurm_job_id=None, node_name=None):
    insert_db_id(node_name=node_name)  # get permission to make a db connection

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
    cand4.slurm_job_id = slurm_job_id
    cand4.node = node_name
    db.session.commit()
    db.session.close()

    remove_db_id(node_name=node_name)  # release permission for this db connection
    return cand_id


def finish_cand(cand_id, node_name):
    insert_db_id(node_name=node_name)  # get permission to make a db connection

    cand = db.session.query(CandidateLevel4).filter(
        CandidateLevel4.id == cand_id).one()
    cand.pspl_gp_fit_finished = True
    cand.pspl_gp_fit_datetime_finished = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id(node_name=node_name)  # release permission for this db connection


def fit_level4_cand_to_pspl_gp(cand_id, node_name=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cand_fitter_data = load_cand_fitter_data(cand_id)
    data = cand_fitter_data['data']
    fitter_params = cand_fitter_data['fitter_params']
    out_dir = cand_fitter_data['out_dir']
    outputfiles_basename = f'{out_dir}/{cand_id}_'

    num_lightcurves = cand_fitter_data['fitter_params']['num_lightcurves']
    n_live_points = int(min(300 * num_lightcurves, 1000))

    fitter = model_fitter.PSPL_Solver(data,
                                      PSPL_Phot_Par_GP_Param2_2,
                                      add_error_on_photometry=False,
                                      multiply_error_on_photometry=False,
                                      use_phot_optional_params=True,
                                      importance_nested_sampling=False,
                                      n_live_points=n_live_points,
                                      evidence_tolerance=0.5,
                                      sampling_efficiency=0.8,
                                      outputfiles_basename=outputfiles_basename)

    # set priors
    fitter.priors['t0'] = model_fitter.make_norm_gen(fitter_params['t0'], fitter_params['tE']*.5)
    u0_amp_std = 0.25
    u0_amp_low = max(fitter_params['u0_amp'] * (1 - u0_amp_std), -1.2)
    u0_amp_high = min(fitter_params['u0_amp'] * (1 + u0_amp_std), 1.2)
    u0_amp_low_sigma = u0_amp_low / u0_amp_std
    u0_amp_high_sigma = u0_amp_high / u0_amp_std
    fitter.priors['u0_amp'] = model_fitter.make_truncnorm_gen(fitter_params['u0_amp'], u0_amp_std,
                                                              u0_amp_low_sigma, u0_amp_high_sigma)
    fitter.priors['tE'] = model_fitter.make_norm_gen(fitter_params['tE'], fitter_params['tE']*.5)
    fitter.priors['piE_E'] = model_fitter.make_norm_gen(fitter_params['piE_E'], .5)
    fitter.priors['piE_N'] = model_fitter.make_norm_gen(fitter_params['piE_N'], .5)

    for idx in range(1, num_lightcurves+1):
        b_sff_std = 0.2
        b_sff_low = max(fitter_params[f'b_sff_{idx}'] * (1 - b_sff_std), 0)
        b_sff_high = min(fitter_params[f'b_sff_{idx}'] * (1 + b_sff_std), 1.2)
        b_sff_low_sigma = b_sff_low / b_sff_std
        b_sff_high_sigma = b_sff_high / b_sff_std
        fitter.priors[f'b_sff{idx}'] = model_fitter.make_truncnorm_gen(fitter_params[f'b_sff_{idx}'], b_sff_std,
                                                                       b_sff_low_sigma, b_sff_high_sigma)
        fitter.priors[f'mag_base{idx}'] = model_fitter.make_norm_gen(fitter_params[f'mag_base_{idx}'], 0.2)
        fitter.priors[f'gp_log_sigma{idx}'] = model_fitter.make_norm_gen(0, 5)
        fitter.priors[f'gp_rho{idx}'] = model_fitter.make_invgamma_gen(data[f't_phot{idx}'])
        fitter.priors[f'gp_log_omega04_S0{idx}'] = model_fitter.make_norm_gen(np.median(data[f'mag_err{idx}']) ** 2, 5)
        fitter.priors[f'gp_log_omega0{idx}'] = model_fitter.make_norm_gen(0, 5)

    comm.Barrier()
    fitter.solve()

    if rank == 0:
        logger.info(f'{cand_id} : Fit complete, now plotting')
        fitter.plot_dynesty_style(fit_vals='maxl')

        best = fitter.get_best_fit(def_best='maxl')
        # model_params = {'t0': best['t0'],
        #                 'u0_amp': best['u0_amp'],
        #                 'tE': best['tE'],
        #                 'piE_E': best['piE_E'],
        #                 'piE_N': best['piE_N'],
        #                 'raL': data['raL'],
        #                 'decL': data['decL']}
        # multi_params = ['b_sff', 'mag_base', 'gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']
        # for param in multi_params:
        #     model_params[param] = [best[f'{param}{i}'] for i in range(1, fitter.n_phot_sets + 1)]
        # pspl_out = PSPL_Phot_Par_GP_Param2_2(**model_params)
        best_model = fitter.get_model(best)
        fitter.plot_model_and_data(best_model)
        logger.info(f'{cand_id} : Plotting complete')

    comm.Barrier()
    if rank == 0:
        finish_cand(cand_id, node_name)


def fit_level4_cands_to_pspl_gp(single_job=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) > 1:
        slurm_job_id = sys.argv[1]
        node_name = sys.argv[2]
    else:
        slurm_job_id = None
        node_name = None

    while True:

        if rank == 0:
            cand_id = fetch_cand(slurm_job_id=slurm_job_id,
                                 node_name=node_name)
            logger.info(f'{cand_id} : Starting fit')
        else:
            cand_id = None
        cand_id = comm.bcast(cand_id, root=0)
        fit_level4_cand_to_pspl_gp(cand_id, node_name)
        comm.Barrier()
        if single_job:
            return


if __name__ == '__main__':
    logging.getLogger('ulensdb').setLevel(logging.WARNING)
    fit_level4_cands_to_pspl_gp()
