#! /usr/bin/env python
"""
plot_process_progress.py
"""

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from puzle.models import SourceIngestJob, StarProcessJob
from puzle.utils import return_figures_dir, return_DR5_dir, return_data_dir
from puzle import db


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def _load_process_progress(type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_progress_progress.dict'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return {}


def load_star_process_progress():
    return _load_process_progress('star')


def _save_process_progress(process_progress, type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_progress_progress.dict'
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pickle.dump(process_progress, f)


def save_star_process_progress(process_progress):
    _save_process_progress(process_progress, 'star')


def plot_star_process_progress():
    DR5_dir = return_DR5_dir()
    jobs = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id).\
        filter(StarProcessJob.finished == True).all()

    star_process_progress = load_star_process_progress()

    N_stars_arr = []
    ra_arr = []
    dec_arr = []
    for i, job in enumerate(jobs):
        if i % 1000 == 0:
            print(i, len(jobs))
        job_id = job[0].id

        if job_id in star_process_progress:
            ra = star_process_progress[job_id][0]
            dec = star_process_progress[job_id][1]
            N_stars = star_process_progress[job_id][2]
        else:
            ra = (job[0].ra_start + job[0].ra_end) / 2
            dec = (job[0].dec_start + job[0].dec_end) / 2
            dir = '%s/stars_%s' % (DR5_dir, str(job_id)[:3])
            fname = f'{dir}/stars.{job_id:06}.txt'
            N_stars = file_len(fname)
            star_process_progress[job_id] = (ra, dec, N_stars)

        N_stars_arr.append(N_stars)
        ra_arr.append(ra)
        dec_arr.append(dec)

    save_star_process_progress(star_process_progress)

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

    fname = '%s/stars_process_progress.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    plot_star_process_progress()
