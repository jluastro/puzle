#! /usr/bin/env python
"""
cands.py
"""

import numpy as np

from puzle.models import Source, Candidate
from puzle import db


def fetch_cand_by_id(cand_id):
    cands = Candidate.query.filter(Candidate.id==cand_id).all()
    if len(cands) == 1:
        cand = cands[0]
    else:
        print('No candidates found.')
        cand = None
    return cand


def fetch_cand_by_radec(ra, dec, radius=2):
    cone_filter = Candidate.cone_search(ra, dec, radius=radius)
    cands = db.session.query(Candidate).filter(cone_filter).all()
    if len(cands) == 1:
        cand = cands[0]
    elif len(cands) > 1:
        print('Multiple cands within return_cand_by_radec. Returning closest.')
        ra_arr = [c.ra for c in cands]
        dec_arr = [c.dec for c in cands]
        dist_arr = np.hypot(ra_arr-ra, dec_arr-dec)
        idx = np.argmin(dist_arr)
        cand = cands[idx]
    else:
        print('No candidates found.')
        cand = None
    return cand


def return_best_obj(cand):
    idx = cand.idx_best
    source_id = cand.source_id_arr[idx]
    source = Source.query.filter(Source.id==source_id).first()
    color = cand.color_arr[idx]
    obj = getattr(source.zort_source, f'object_{color}')
    return obj


def fetch_cand_best_obj_by_id(cand_id):
    cand = fetch_cand_by_id(cand_id)
    if cand is None:
        obj = None
    else:
        obj = return_best_obj(cand)
    return obj
