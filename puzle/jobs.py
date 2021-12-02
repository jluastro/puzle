#! /usr/bin/env python
"""
jobs.py
"""

import os
import pickle
import numpy as np

from puzle.models import SourceIngestJob, StarProcessJob
from puzle.utils import return_DR5_dir, return_data_dir
from puzle import db


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def _load_process_progress(type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_process_progress.dict'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return {}


def load_star_process_progress():
    return _load_process_progress('star')


def _save_process_progress(process_progress, type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_process_progress.dict'
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pickle.dump(process_progress, f)


def save_star_process_progress(process_progress):
    _save_process_progress(process_progress, 'star')


def return_num_objs_arr():
    DR5_dir = return_DR5_dir()
    jobs = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id).\
        all()

    star_process_progress = load_star_process_progress()

    num_stars_arr = []
    ra_arr = []
    dec_arr = []
    num_objs_arr = []
    datetime_delta_arr = []
    for i, job in enumerate(jobs):
        if i % 1000 == 0:
            print(i, len(jobs))
        job_id = job[0].id
        if job[1].num_objs is None:
            num_objs = 0
        else:
            num_objs = job[1].num_objs

        if job_id in star_process_progress:
            ra = star_process_progress[job_id][0]
            dec = star_process_progress[job_id][1]
            num_stars = star_process_progress[job_id][2]
        else:
            ra = (job[0].ra_start + job[0].ra_end) / 2
            dec = (job[0].dec_start + job[0].dec_end) / 2
            dir = '%s/stars_%s' % (DR5_dir, str(job_id)[:3])
            fname = f'{dir}/stars.{job_id:06}.txt'
            num_stars = file_len(fname)
            star_process_progress[job_id] = (ra, dec, num_stars)

        datetime_delta = job[1].datetime_finished - job[1].datetime_started
        datetime_delta_arr.append(datetime_delta.seconds)

        num_stars_arr.append(num_stars)
        num_objs_arr.append(num_objs)
        ra_arr.append(ra)
        dec_arr.append(dec)

    save_star_process_progress(star_process_progress)

    ra_arr = np.array(ra_arr)
    dec_arr = np.array(dec_arr)
    num_objs_arr = np.array(num_objs_arr)

    return ra_arr, dec_arr, num_objs_arr
