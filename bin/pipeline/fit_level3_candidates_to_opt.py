#! /usr/bin/env python
"""
fit_level3_candidates_to_opt.py
"""

import os
import numpy as np

from puzle.cands import fit_cand_id_to_opt
from puzle.models import CandidateLevel3


def fit_level3_candidates_to_ulens():
    cand_ids = [c[0] for c in CandidateLevel3.query.order_by(CandidateLevel3.id).\
                                    filter(CandidateLevel3.eta_best==None).\
                                    with_entities(CandidateLevel3.id).all()]

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cand_ids = np.array_split(cand_ids, size)[rank]
    print('Rank %i) %i candidates to fit' % (rank, len(my_cand_ids)))

    for cand_id in my_cand_ids:
        fit_cand_id_to_opt(cand_id, uploadFlag=True, plotFlag=False)


if __name__ == '__main__':
    fit_level3_candidates_to_ulens()
