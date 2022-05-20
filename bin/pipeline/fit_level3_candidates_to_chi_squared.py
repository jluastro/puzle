#! /usr/bin/env python
"""
fit_level3_candidates_to_chi_squared.py
"""

import os
import numpy as np

from puzle.stats import calculate_chi_squared_inside_outside
from puzle.cands import fetch_cand_best_obj_by_id
from puzle.models import CandidateLevel3
from puzle import db


def fit_level3_candidates_to_chi_squared():
    cands = CandidateLevel3.query.order_by(CandidateLevel3.id).\
                                    with_entities(CandidateLevel3.id,
                                                  CandidateLevel3.t0_best,
                                                  CandidateLevel3.tE_best).\
                                    filter(CandidateLevel3.t0_best!=0,
                                           CandidateLevel3.chi_squared_flat_inside_1tE_best==None,
                                           CandidateLevel3.idx_best!=-99).all()
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
        obj = fetch_cand_best_obj_by_id(cand_id)
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr

        t0 = t0_arr[idx]
        tE = tE_arr[idx]

        update_dct = {}
        for tE_factor in [1, 2, 3]:
            data = calculate_chi_squared_inside_outside(hmjd, mag, magerr, t0, tE, tE_factor)
            chi_squared_inside, chi_squared_outside, num_days_inside, num_days_outside, delta_hmjd_outside = data
            key = f'{tE_factor}tE_best'
            update_dct[f'chi_squared_flat_inside_{key}'] = chi_squared_inside
            update_dct[f'chi_squared_flat_outside_{key}'] = chi_squared_outside
            update_dct[f'num_days_inside_{key}'] = int(num_days_inside)
            update_dct[f'num_days_outside_{key}'] = int(num_days_outside)
            update_dct[f'delta_hmjd_outside_{key}'] = delta_hmjd_outside

        db.session.query(CandidateLevel3).filter(CandidateLevel3.id==cand_id).update(update_dct)
    db.session.commit()


if __name__ == '__main__':
    fit_level3_candidates_to_chi_squared()
