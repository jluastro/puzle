#! /usr/bin/env python
"""
upload_level4_candidates_pspl_gp_fit.py
"""

import numpy as np
import os

from puzle.pspl_gp_fit import upload_pspl_gp_results_by_cand_id
from puzle.models import CandidateLevel4
from puzle import db


def upload_level4_candidates_pspl_gp_fit(def_best='map'):
    cands = db.session.query(CandidateLevel4). \
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp == None). \
        with_entities(CandidateLevel4.id). \
        order_by(CandidateLevel4.id). \
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
    for cand_id in my_cand_ids:
        upload_pspl_gp_results_by_cand_id(cand_id, def_best=def_best, recomputeFlag=False)


if __name__ == '__main__':
    upload_level4_candidates_pspl_gp_fit(def_best='map')
