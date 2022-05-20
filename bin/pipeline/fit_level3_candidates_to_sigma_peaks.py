#! /usr/bin/env python
"""
fit_level3_candidates_to_sigma_peaks.py
"""

import os
import numpy as np

from puzle.cands import fetch_cand_best_obj_by_id, return_sigma_peaks
from puzle.models import CandidateLevel3
from puzle import db


def fit_level3_candidates_to_sigma_peaks():
    cands = CandidateLevel3.query.order_by(CandidateLevel3.id).\
                                    with_entities(CandidateLevel3.id,
                                                  CandidateLevel3.t0_best,
                                                  CandidateLevel3.tE_best).\
                                    filter(CandidateLevel3.t0_best!=0,
                                           CandidateLevel3.num_3sigma_peaks_inside_2tE_best==None,
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

        t0 = t0_arr[idx]
        tE = tE_arr[idx]

        update_dct = {}
        for sigma_factor in [3, 5]:
            sigma_peaks_inside, sigma_peaks_outside = return_sigma_peaks(hmjd, mag, t0, tE,
                                                                         sigma_factor=sigma_factor,
                                                                         tE_factor=2)
            update_dct[f'num_{sigma_factor}sigma_peaks_inside_2tE_best'] = int(sigma_peaks_inside)
            update_dct[f'num_{sigma_factor}sigma_peaks_outside_2tE_best'] = int(sigma_peaks_outside)
        db.session.query(CandidateLevel3).filter(CandidateLevel3.id==cand_id).update(update_dct)
    db.session.commit()


if __name__ == '__main__':
    fit_level3_candidates_to_sigma_peaks()
