#! /usr/bin/env python
"""
plot_ingest_progress.py
"""

import matplotlib.pyplot as plt
import numpy as np
from puzle.models import SourceIngestJob, StarIngestJob
from puzle.utils import return_figures_dir, return_DR3_dir
from puzle import db


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def plot_source_ingest_progress():
    DR3_dir = return_DR3_dir()
    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.finished == True).all()

    N_sources_arr = []
    ra_arr = []
    dec_arr = []
    for i, job in enumerate(jobs):
        if i % 100 == 0:
            print(i, len(jobs))
        job_id = job.id
        dir = '%s/sources_%s' % (DR3_dir, str(job_id)[:3])
        fname = f'{dir}/sources.{job_id:06}.txt'
        N_sources = file_len(fname)

        ra = (job.ra_start + job.ra_end) / 2
        dec = (job.dec_start + job.dec_end) / 2

        N_sources_arr.append(N_sources)
        ra_arr.append(ra)
        dec_arr.append(dec)
    ra_arr = np.array(ra_arr)
    dec_arr = np.array(dec_arr)
    N_sources_arr = np.array(N_sources_arr)

    N_sources_arr = np.log10(N_sources_arr)
    N_sources_arr[np.isinf(N_sources_arr)] = 0

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.scatter(ra_arr, dec_arr, c=N_sources_arr, edgecolor='None', s=5)
    cond = N_sources_arr == 0
    ax.scatter(ra_arr[cond], dec_arr[cond], c='r', edgecolor='None', s=5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log(num sources)', fontsize=12)
    ax.set_xlabel('ra', fontsize=12)
    ax.set_ylabel('dec', fontsize=12)
    fig.tight_layout()

    fname = '%s/sources_ingest_progress.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)


def plot_star_ingest_progress():
    DR3_dir = return_DR3_dir()
    jobs = db.session.query(SourceIngestJob, StarIngestJob).\
        filter(SourceIngestJob.id == StarIngestJob.source_ingest_job_id).\
        filter(StarIngestJob.finished == True).all()

    N_stars_arr = []
    ra_arr = []
    dec_arr = []
    for i, job in enumerate(jobs):
        if i % 100 == 0:
            print(i, len(jobs))
        job_id = job[0].id
        dir = '%s/stars_%s' % (DR3_dir, str(job_id)[:3])
        fname = f'{dir}/stars.{job_id:06}.txt'
        N_stars = file_len(fname)

        ra = (job[0].ra_start + job[0].ra_end) / 2
        dec = (job[0].dec_start + job[0].dec_end) / 2

        N_stars_arr.append(N_stars)
        ra_arr.append(ra)
        dec_arr.append(dec)
    ra_arr = np.array(ra_arr)
    dec_arr = np.array(dec_arr)
    N_stars_arr = np.array(N_stars_arr)

    N_stars_arr = np.log10(N_stars_arr)
    N_stars_arr[np.isinf(N_stars_arr)] = 0

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.scatter(ra_arr, dec_arr, c=N_stars_arr, edgecolor='None', s=5)
    cond = N_stars_arr == 0
    ax.scatter(ra_arr[cond], dec_arr[cond], c='r', edgecolor='None', s=5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log(num stars)', fontsize=12)
    ax.set_xlabel('ra', fontsize=12)
    ax.set_ylabel('dec', fontsize=12)
    fig.tight_layout()

    fname = '%s/stars_ingest_progress.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    plot_source_ingest_progress()
    plot_star_ingest_progress()
