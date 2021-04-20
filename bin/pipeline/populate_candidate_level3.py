#! /usr/bin/env python
"""
populate_candidate_level3.py
"""

from puzle.models import CandidateLevel2, CandidateLevel3
from puzle.cands import apply_eta_residual_slope_offset_to_query
from puzle import db


def populate_candidate_level3():
    cands_level2 = apply_eta_residual_slope_offset_to_query(CandidateLevel2.query).all()
    print('%i candidates identified' % len(cands_level2))

    for cand_level2 in cands_level2:
        cand_level3 = CandidateLevel3(id=cand_level2.id,
                                      ra=cand_level2.ra,
                                      dec=cand_level2.dec,
                                      source_id_arr=cand_level2.source_id_arr,
                                      color_arr=cand_level2.color_arr,
                                      pass_arr=cand_level2.pass_arr,
                                      idx_best=cand_level2.idx_best,
                                      num_objs_pass=cand_level2.num_objs_pass,
                                      num_objs_tot=cand_level2.num_objs_tot)
        db.session.add(cand_level3)
    db.session.commit()
    db.session.close()

    print('Upload to candidate_level3 complete')


if __name__ == '__main__':
    populate_candidate_level3()
