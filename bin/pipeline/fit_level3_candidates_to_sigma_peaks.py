#! /usr/bin/env python
"""
fit_level3_candidates_to_sigma_peaks.py
"""

import os
from astropy.stats import sigma_clip
import numpy as np

from puzle.cands import fetch_cand_best_obj_by_id
from puzle.models import CandidateLevel3
from puzle import db


def return_peak_cond(hmjd, mag, t0, tE, sigma_factor=3, tE_factor=2):
    # Mask out those points that are within the microlensing event
    ulens_mask = hmjd > t0 - tE_factor * tE
    ulens_mask *= hmjd < t0 + tE_factor * tE
    # Use this mask to generated a masked array for sigma clipping
    # By applying this mask, the 3-sigma will not be calculated using these points
    mag_masked = np.ma.array(mag, mask=ulens_mask)
    # Perform the sigma clipping
    mag_masked = sigma_clip(mag_masked, sigma=3, maxiters=5)
    # This masked array is now a mask that includes BOTH the mirolensing event and
    # the 3-sigma outliers that we want removed. This is the "flats" where we
    # want to calculate the mean and sigma
    mean_flat = mag_masked.mean()
    std_flat = mag_masked.std()
    # We now add up the number of sigma peaks within tE
    ulens_cond = ~ulens_mask
    sigma_peaks = np.sum(mag[ulens_cond] <= mean_flat - sigma_factor * std_flat)
    return sigma_peaks


def fit_level3_candidates_to_sigma_peaks():
    cands = CandidateLevel3.query.order_by(CandidateLevel3.id).\
                                    with_entities(CandidateLevel3.id,
                                                  CandidateLevel3.t0_best,
                                                  CandidateLevel3.tE_best).\
                                    filter(CandidateLevel3.t0_best!=0,
                                           CandidateLevel3.num_3sigma_peaks_inside_2tE_best==None).all()
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
            sigma_peaks = return_peak_cond(hmjd, mag, t0, tE,
                                           sigma_factor=sigma_factor, tE_factor=2)
            update_dct[f'num_{sigma_factor}sigma_peaks_inside_2tE_best'] = sigma_peaks

        db.session.query(CandidateLevel3).filter(CandidateLevel3.id==cand_id).update(update_dct)
    db.session.commit()


if __name__ == '__main__':
    fit_level3_candidates_to_sigma_peaks()
