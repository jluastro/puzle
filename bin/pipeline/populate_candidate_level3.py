#! /usr/bin/env python
"""
populate_candidate_level3.py
"""
import os
import numpy as np

from puzle.models import CandidateLevel2, CandidateLevel3
from puzle.cands import apply_eta_residual_slope_offset_to_query, fetch_cand_best_obj_by_id
from puzle import db


def populate_candidate_level3():
    query = CandidateLevel2.query.order_by(CandidateLevel2.id)
    cands_level2 = apply_eta_residual_slope_offset_to_query(query).all()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cands_level2 = np.array_split(cands_level2, size)[rank]

    for cand_level2 in my_cands_level2:
        obj = fetch_cand_best_obj_by_id(cand_level2.id)
        num_epochs = obj.lightcurve.nepochs
        num_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
        cand_level3 = CandidateLevel3(id=cand_level2.id,
                                      ra=cand_level2.ra,
                                      dec=cand_level2.dec,
                                      source_id_arr=cand_level2.source_id_arr,
                                      color_arr=cand_level2.color_arr,
                                      pass_arr=cand_level2.pass_arr,
                                      idx_best=cand_level2.idx_best,
                                      num_objs_pass=cand_level2.num_objs_pass,
                                      num_objs_tot=cand_level2.num_objs_tot,
                                      num_epochs_best=num_epochs,
                                      num_days_best=num_days)
        db.session.add(cand_level3)
        del obj
    db.session.commit()
    db.session.close()

    print('Upload to candidate_level3 complete')


if __name__ == '__main__':
    populate_candidate_level3()
