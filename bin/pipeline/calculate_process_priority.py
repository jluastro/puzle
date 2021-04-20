#! /usr/bin/env python
"""
calculate_process_priority.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from puzle.models import SourceIngestJob, StarProcessJob
from puzle import db


def calculate_process_priority():
    jobs = db.session.query(SourceIngestJob, StarProcessJob). \
        outerjoin(SourceIngestJob, StarProcessJob.source_ingest_job_id == SourceIngestJob.id). \
        all()

    # find the density at the center of the job field
    ra_arr = []
    dec_arr = []
    for source_ingest_job, _ in jobs:
        ra = (source_ingest_job.ra_start + source_ingest_job.ra_end) / 2.
        dec = (source_ingest_job.dec_start + source_ingest_job.dec_end) / 2.
        ra_arr.append(ra)
        dec_arr.append(dec)
    ra_arr = np.array(ra_arr)
    dec_arr = np.array(dec_arr)

    # condition in galactic latitude
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    l_arr = coords.galactic.l.value
    l_arr[l_arr > 180] -= 360
    b_arr = coords.galactic.b.value
    cond_gal = np.abs(b_arr) <= 20
    dist_arr = np.hypot(l_arr, b_arr)
    #
    # most to least dense
    dist_offest_arr = np.zeros(len(dist_arr))
    dist_offest_arr[:] = dist_arr
    dist_offest_arr[~cond_gal] += np.max(dist_arr[cond_gal])

    dist_offest_idx_arr = np.argsort(dist_offest_arr)

    # add priority to star_process_job
    for i, (_, star_process_job) in enumerate(jobs):
        priority = int(np.where(dist_offest_idx_arr == i)[0][0])
        star_process_job.priority = priority
        db.session.add(star_process_job)
    db.session.commit()
    db.session.close()


def plot_process_priority():
    jobs = db.session.query(SourceIngestJob, StarProcessJob). \
        outerjoin(SourceIngestJob, StarProcessJob.source_ingest_job_id == SourceIngestJob.id). \
        all()

    ra_arr = []
    dec_arr = []
    priority_arr = []
    for source_ingest_job, star_process_job in jobs:
        ra = (source_ingest_job.ra_start + source_ingest_job.ra_end) / 2.
        dec = (source_ingest_job.dec_start + source_ingest_job.dec_end) / 2.
        priority = star_process_job.priority
        ra_arr.append(ra)
        dec_arr.append(dec)
        priority_arr.append(priority)
    priority_arr = np.array(priority_arr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    im = ax.scatter(ra_arr, dec_arr,
                    c=priority_arr,
                    s=1, alpha=1)
    fig.colorbar(im, ax=ax, label='priority')
    ax.set_xlabel('ra', fontsize=12)
    ax.set_ylabel('dec', fontsize=12)
    ax.set_title('Star Process Job Ordering', fontsize=12)
    fig.tight_layout()


if __name__ == '__main__':
    calculate_process_priority()
