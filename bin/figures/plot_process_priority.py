#! /usr/bin/env python
"""
plot_process_priority.py
"""

import matplotlib.pyplot as plt
from puzle.models import SourceIngestJob, StarProcessJob
from puzle.utils import return_figures_dir
from puzle import db


def plot_star_process_priority():
    jobs = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id).\
        all()

    priority_arr = []
    ra_arr = []
    dec_arr = []
    for i, job in enumerate(jobs):
        ra = (job[0].ra_start + job[0].ra_end) / 2
        dec = (job[0].dec_start + job[0].dec_end) / 2
        priority = job[1].priority

        priority_arr.append(priority)
        ra_arr.append(ra)
        dec_arr.append(dec)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.scatter(ra_arr, dec_arr, c=priority_arr, edgecolor='None', s=3)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('priority', fontsize=12)
    ax.set_xlabel('ra', fontsize=12)
    ax.set_ylabel('dec', fontsize=12)
    fig.tight_layout()

    fname = '%s/stars_process_priority.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


if __name__ == '__main__':
    plot_star_process_priority()
