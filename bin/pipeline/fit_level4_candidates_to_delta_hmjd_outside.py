#! /usr/bin/env python
"""
fit_level4_candidates_to_delta_hmjd_outside.py
"""

import os
import numpy as np

from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.models import CandidateLevel4
from puzle import db


def fit_level4_candidates_to_delta_hmjd_outside():
    cands = CandidateLevel4.query.order_by(CandidateLevel4.id).\
                                    with_entities(CandidateLevel4.id,
                                                  CandidateLevel4.t0_pspl_gp,
                                                  CandidateLevel4.tE_pspl_gp).\
                                    filter(CandidateLevel4.fit_type_pspl_gp != None,
                                           CandidateLevel4.delta_hmjd_outside_pspl_gp != None).all()
    num_cands = len(cands)
    cand_id_arr = [c[0] for c in cands]
    t0_arr = [c[1] for c in cands]
    tE_arr = [c[2] for c in cands]

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_idx_arr = np.array_split(np.arange(num_cands), size)[rank]
    print('Rank %i) %i candidates to fit' % (rank, len(my_idx_arr)))

    for idx in my_idx_arr:
        cand_id = cand_id_arr[idx]
        t0 = t0_arr[idx]
        tE = tE_arr[idx]

        cand_fitter_data = load_cand_fitter_data(cand_id)
        num_lightcurves = cand_fitter_data['fitter_params']['num_lightcurves']
        data = cand_fitter_data['data']

        delta_hmjd_outside_max = 0
        for i in range(1, num_lightcurves+1):
            hmjd = data[f't_phot{i}']

            cond_low = hmjd < t0 - 2 * tE
            if np.sum(cond_low) > 1:
                delta_hmjd_low = np.max(hmjd[cond_low]) - np.min(hmjd[cond_low])
            else:
                delta_hmjd_low = 0

            cond_high = hmjd > t0 + 2 * tE
            if np.sum(cond_high) > 1:
                delta_hmjd_high = np.max(hmjd[cond_high]) - np.min(hmjd[cond_high])
            else:
                delta_hmjd_high = 0
            delta_hmjd_outside = delta_hmjd_low + delta_hmjd_high
            delta_hmjd_outside_max = max(delta_hmjd_outside, delta_hmjd_outside_max)

        update_dct = {'delta_hmjd_outside_pspl_gp': delta_hmjd_outside_max}

        db.session.query(CandidateLevel4).\
            filter(CandidateLevel4.id==cand_id).\
            update(update_dct)
    db.session.commit()


if __name__ == '__main__':
    fit_level4_candidates_to_delta_hmjd_outside()
