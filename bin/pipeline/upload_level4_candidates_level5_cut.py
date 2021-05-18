#! /usr/bin/env python
"""
upload_level4_candidates_level5_cut.py
"""

import numpy as np
import os

from puzle.models import CandidateLevel4
from puzle import db


def upload_level4_candidates_level5_cut():
    cands = db.session.query(CandidateLevel4). \
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None). \
        all()
    level5_arr = np.ones(len(cands)).astype(bool)

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cands = np.array_split(cands, size)[rank]
    my_level5_arr = np.array_split(level5_arr, size)[rank]
    for cand, level5 in zip(my_cands, my_level5_arr):
        cand.level5 = level5
    db.session.commit()


if __name__ == '__main__':
    upload_level4_candidates_level5_cut()
