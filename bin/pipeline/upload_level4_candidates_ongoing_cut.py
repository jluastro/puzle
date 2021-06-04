#! /usr/bin/env python
"""
upload_level4_candidates_ongoing_cut.py
"""

import numpy as np

from puzle.utils import MJD_finish
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


def upload_level4_candidates_ongoing_cut():
    cands = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
                                                        CandidateLevel4.id == CandidateLevel3.id). \
        filter(CandidateLevel3.t0_best + CandidateLevel3.tE_best >= MJD_finish,
               CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        all()

    keys = ['t0',
            't0_err'
            'tE',
            'tE_err',
            'rchi2']
    keys_err = [k for k in keys if f'{k}_err' in keys]

    data = {}
    for key in keys:
        data[key] = np.array([getattr(c, f'{key}_pspl_gp') for c in cands])
    data['cand_id'] = np.array([c.id for c in cands])

    error_frac = {}
    for key in keys_err:
        key_err = f'{key}_err'
        error_frac[key] = data[key_err] / data[key]

    cond1 = error_frac['t0'] <= 0.8
    cond2 = error_frac['tE'] <= 0.8
    cond3 = data['rchi2'] <= 3

    ongoing_cond = cond1 * cond2 * cond3

    print('No filters', len(cond1), 'cands')
    print('Filters up to 1', np.sum(cond1), 'cands')
    print('Filters up to 2', np.sum(cond1 * cond2), 'cands')
    print('Filters up to 3', np.sum(cond1 * cond2 * cond3), 'cands')

    for i, (cand, ongoing_cond) in enumerate(zip(cands, ongoing_cond)):
        cand.ongoing = ongoing_cond
    db.session.commit()


if __name__ == '__main__':
    upload_level4_candidates_ongoing_cut()
