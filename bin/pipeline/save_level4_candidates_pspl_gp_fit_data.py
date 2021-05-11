#! /usr/bin/env python
"""
save_level4_candidates_pspl_gp_fit_data.py
"""

import os
import numpy as np

from puzle.pspl_gp_fit import save_cand_fitter_data_by_id
from puzle.models import CandidateLevel4, CandidateLevel3
from puzle.utils import MJD_finish
from puzle import db


def save_level4_candidates_pspl_gp_fit_data():
    cands = db.session.query(CandidateLevel3, CandidateLevel4).\
        filter(CandidateLevel3.id == CandidateLevel4.id,
               CandidateLevel3.t0_best + CandidateLevel3.tE_best >= MJD_finish).\
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


if __name__ == '__main__':
    save_level4_candidates_pspl_gp_fit_data()