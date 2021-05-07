#! /usr/bin/env python
"""
fit_level4_candidates_to_pspl_gp.py
"""

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
    t0 = cand3.t0_best
    tE = cand4.tE_best
    cand4.pspl_gp_fit_started = True
    cand4.pspl_gp_fit_datetime_started = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection
    return cand_id, t0, tE


def finish_cand(cand_id):
    insert_db_id()  # get permission to make a db connection

    cand = db.session.query(CandidateLevel4).filter(
        CandidateLevel4.id == cand_id).one()
    cand.pspl_gp_fit_finished = True
    cand.pspl_gp_fit_datetime_finished = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def fit_level4_cand_to_pspl_gp(cand_id, t0, tE):
    cand_fitter_data = load_cand_fitter_data(cand_id)
    data = cand_fitter_data['data']
    phot_priors = cand_fitter_data['phot_priors']
    out_dir = cand_fitter_data['out_dir']
    idx_data_best = phot_priors['idx_data_best']
    outputfiles_basename = f'{out_dir}/fit_'

    fitter = model_fitter.PSPL_Solver(data,
                                      PSPL_Phot_Par_GP_Param2_2,
                                      multiply_error_on_photometry=True,
                                      n_live_points=400,
                                      evidence_tolerance=0.01,
                                      outputfiles_basename=outputfiles_basename)

    fitter.priors['t0'] = model_fitter.make_norm_gen(t0, 100)
    fitter.priors['u0_amp'] = model_fitter.make_gen(-1.2, 1.2)
    fitter.priors['tE'] = model_fitter.make_norm_gen(tE, 50)
    fitter.priors['piE_E'] = model_fitter.make_gen(-1, 1)
    fitter.priors['piE_N'] = model_fitter.make_gen(-1, 1)
    fitter.priors['gp_log_sigma'] = model_fitter.make_norm_gen(0, 5)
    fitter.priors['gp_rho'] = model_fitter.make_invgamma_gen(data[f't_phot{idx_data_best}'])
    fitter.priors['gp_log_S0'] = model_fitter.make_norm_gen(0, 5)
    fitter.priors['gp_log_omega0'] = model_fitter.make_norm_gen(0, 5)

    for k, v in phot_priors.items():
        fitter.priors[k] = v

    fitter.solve()

    best = fitter.get_best_fit(def_best='maxl')
    pspl_out = PSPL_Phot_Par_GP_Param2_2(t0=best['t0'],
                                         u0_amp=best['u0_amp'],
                                         tE=best['tE'],
                                         piE_E=best['piE_E'],
                                         piE_N=best['piE_N'],
                                         b_sff=best[f'b_sff{idx_data_best}'],
                                         mag_src=best[f'mag_src{idx_data_best}'],
                                         gp_log_sigma=best[f'gp_log_sigma{idx_data_best}'],
                                         gp_log_rho=best[f'gp_log_rho{idx_data_best}'],
                                         gp_log_S0=best[f'gp_log_S0{idx_data_best}'],
                                         gp_log_omega0=best[f'gp_log_omega0{idx_data_best}'],
                                         raL=data['raL'],
                                         decL=data['decL'])

    fitter.plot_dynesty_style(fit_vals='maxl')
    fitter.plot_model_and_data(pspl_out)

    finish_cand(cand_id)


def fit_level4_cands_to_pspl_gp(single_job=False):

    while True:
        cand_id, t0, tE = fetch_cand()
        fit_level4_cand_to_pspl_gp(cand_id, t0, tE)

        if single_job:
            return


if __name__ == '__main__':
    fit_level4_cands_to_pspl_gp()
