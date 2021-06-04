#! /usr/bin/env python
"""
upload_level4_candidates_level5_cut.py
"""

import numpy as np

from puzle.utils import MJD_finish
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


def upload_level4_candidates_level5_cut():
    cands = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
                                                        CandidateLevel4.id == CandidateLevel3.id). \
        filter(CandidateLevel3.t0_best + CandidateLevel3.tE_best < MJD_finish,
               CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        all()

    keys = ['t0',
            'u0_amp',
            'u0_amp_err',
            'tE',
            'tE_err',
            'b_sff',
            'b_sff_err',
            'piE_E',
            'piE_E_err',
            'piE_N',
            'piE_N_err',
            'piE',
            'piE_err',
            'rchi2',
            'delta_hmjd_outside']
    keys_err = [k for k in keys if f'{k}_err' in keys]

    data = {}
    for key in keys:
        data[key] = np.array([getattr(c, f'{key}_pspl_gp') for c in cands])
    data['cand_id'] = np.array([c.id for c in cands])

    error_frac = {}
    for key in keys_err:
        key_err = f'{key}_err'
        error_frac[key] = data[key_err] / data[key]

    cond1 = error_frac['tE'] <= 0.2
    cond2 = np.abs(data['u0_amp']) <= 1.0
    cond3 = data['b_sff'] <= 1.2
    cond4 = data['rchi2'] <= 3
    cond5 = data['delta_hmjd_outside'] / data['tE'] >= 2

    level5_cond = cond1 * cond2 * cond3 * cond4 * cond5

    print('No filters', len(cond1), 'cands')
    print('Filters up to 1', np.sum(cond1), 'cands')
    print('Filters up to 2', np.sum(cond1 * cond2), 'cands')
    print('Filters up to 3', np.sum(cond1 * cond2 * cond3), 'cands')
    print('Filters up to 4', np.sum(cond1 * cond2 * cond3 * cond4), 'cands')
    print('Filters up to 5', np.sum(cond1 * cond2 * cond3 * cond4 * cond5), 'cands')

    for i, (cand, level5_cond) in enumerate(zip(cands, level5_cond)):
        cand.level5 = level5_cond
    db.session.commit()


if __name__ == '__main__':
    upload_level4_candidates_level5_cut()
