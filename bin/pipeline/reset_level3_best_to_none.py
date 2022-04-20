#! /usr/bin/env python
"""
populate_candidate_level3.py
"""
import os
import numpy as np

from puzle.models import CandidateLevel3
from puzle.cands import fetch_cand_best_obj_by_id
from puzle import db


def reset_level3_best_to_none():
    query = CandidateLevel3.query
    cands_level3 = db.session.query(CandidateLevel3).all()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    if rank == 0:
        print('rank 0) %i total candidates identified' % len(cands_level3))

#    cands_level3.eta_best = None
#    cands_level3.chi_squared_flat_inside_1tE_best = None
    
    for cand in cands_level3:
        cand.eta_best = None
        cand.chi_squared_flat_inside_1tE_best = None
        
    db.session.commit()
    db.session.close()

    if size > 1:
        comm.Barrier()
    if rank == 0:
        print('Upload to candidate_level3 complete')

if __name__ == '__main__':
    reset_level3_best_to_none()
