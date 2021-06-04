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
import matplotlib.pyplot as plt

from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.utils import MJD_start, MJD_finish, get_logger
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


logger = get_logger(__name__)


def fetch_cand(slurm_job_id=None, node_name=None):
    insert_db_id(node_name=node_name)  # get permission to make a db connection

    db.session.execute('LOCK TABLE candidate_level3, candidate_level4 '
                       'IN ROW EXCLUSIVE MODE;')
    cand = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
                                                       CandidateLevel4.id == CandidateLevel3.id). \
        filter(CandidateLevel4.pspl_gp_fit_started == False). \
        order_by(CandidateLevel4.num_pspl_gp_fit_lightcurves, func.random()). \
        with_for_update(). \
        first()

    if cand is not None:
        cand_id = cand.id
        cand.pspl_gp_fit_started = True
        cand.pspl_gp_fit_datetime_started = datetime.now()
        cand.slurm_job_id = slurm_job_id
        cand.node = node_name
        db.session.commit()
    else:
        cand_id = None
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
    n_live_points = int(min(300 * num_lightcurves, 1200))

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
    fitter.priors['t0'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params['t0'], fitter_params['tE']*.5,
                                                                      MJD_start - 365, MJD_finish + 365)
    fitter.priors['tE'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params['tE'], fitter_params['tE']*.5,
                                                                      1, 2000)
    fitter.priors['u0_amp'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params['u0_amp'], 0.25,
                                                                          -1.2, 1.2)
    fitter.priors['piE_E'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params['piE_E'], 0.5,
                                                                         -2, 2)
    fitter.priors['piE_N'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params['piE_N'], 0.5,
                                                                         -2, 2)

    # multi parameters
    for idx in range(1, num_lightcurves+1):
        # b_sff / mag_base
        fitter.priors[f'b_sff_{idx}'] = model_fitter.make_truncnorm_gen_with_bounds(fitter_params[f'b_sff_{idx}'], 0.2,
                                                                                    0, 1.2)
        fitter.priors[f'mag_base{idx}'] = model_fitter.make_norm_gen(fitter_params[f'mag_base_{idx}'], 0.2)

        # gp
        fitter.priors[f'gp_log_sigma{idx}'] = model_fitter.make_norm_gen(0, 5)
        fitter.priors[f'gp_rho{idx}'] = model_fitter.make_invgamma_gen(data[f't_phot{idx}'])
        fitter.priors[f'gp_log_omega04_S0{idx}'] = model_fitter.make_norm_gen(np.median(data[f'mag_err{idx}']) ** 2, 5)
        fitter.priors[f'gp_log_omega0{idx}'] = model_fitter.make_norm_gen(0, 5)

    comm.Barrier()
    fitter.solve()

    if rank == 0:
        logger.info(f'{cand_id} : Fit complete, now plotting')
        fitter.plot_dynesty_style(fit_vals='maxl',
                                  traceplot=True, cornerplot=False)

        best = fitter.get_best_fit(def_best='maxl')
        best_model = fitter.get_model(best)
        fitter.plot_model_and_data(best_model)
        plt.close('all')
        logger.info(f'{cand_id} : Plotting complete')

    comm.Barrier()
    if rank == 0:
        finish_cand(cand_id, node_name)
        logger.info(f'{cand_id} : Finished on node {node_name}')


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
            logger.info(f'{cand_id} : Starting fit on {node_name}')
        else:
            cand_id = None
        cand_id = comm.bcast(cand_id, root=0)
        if cand_id is None:
            logger.info(f'{node_name} : Complete!')
            return
        fit_level4_cand_to_pspl_gp(cand_id, node_name)
        comm.Barrier()
        if single_job:
            return


if __name__ == '__main__':
    logging.getLogger('ulensdb').setLevel(logging.WARNING)
    fit_level4_cands_to_pspl_gp()
