#! /usr/bin/env python
"""
upload_level4_candidateS_num_lightcurves.py
"""

import os
import numpy as np

from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.models import CandidateLevel4
from puzle import db


def upload_level4_candidates_num_lightcurves():
    cands = CandidateLevel4.query.order_by(CandidateLevel4.id).all()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cands = np.array_split(cands, size)[rank]
    my_num_cands = len(my_cands)
    print('Rank %i) %i candidates to upload num_lightcurves' % (rank, my_num_cands))

    for cand in my_cands:
        cand_fitter_data = load_cand_fitter_data(cand.id)
        num_lightcurves = cand_fitter_data['fitter_params']['num_lightcurves']
        cand.num_pspl_gp_fit_lightcurves = num_lightcurves
        db.session.add(cand)
    db.session.commit()


if __name__ == '__main__':
    upload_level4_candidates_num_lightcurves()
