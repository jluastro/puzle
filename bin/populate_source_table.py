#! /usr/bin/env python
"""
populate_source_table.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from zort.lightcurveFile import LightcurveFile

from puzle.models import Source, SourceIngestJob
from puzle.utils import fetch_job_enddate
from puzle import db


def convert_obj_to_source(obj, lightcurve_filename):

    filter_dict = {
        1: 'g',
        2: 'r',
        3: 'i'
    }

    source_dict = {
        'object_id_g': None,
        'object_id_r': None,
        'object_id_i': None,
        'lightcurve_position_g': None,
        'lightcurve_position_r': None,
        'lightcurve_position_i': None,
        'lightcurve_filename': lightcurve_filename
    }

    obj_filt = filter_dict[obj.filterid]
    source_dict[f'object_id_{obj_filt}'] = obj.objectid
    source_dict[f'lightcurve_position_{obj_filt}'] = obj.lightcurve_position

    ra_arr = [obj.ra]
    dec_arr = [obj.dec]

    for sib in obj.siblings:
        sib_filt = filter_dict[sib.filterid]
        source_dict[f'object_id_{sib_filt}'] = sib.objectid
        source_dict[f'lightcurve_position_{sib_filt}'] = sib.lightcurve_position
        ra_arr.append(sib.ra)
        dec_arr.append(sib.dec)

    source_dict['ra'] = np.mean(ra_arr)
    source_dict['dec'] = np.mean(dec_arr)

    return Source(**source_dict)


def fetch_job():
    job = db.session.query(SourceIngestJob).\
        filter(SourceIngestJob.started==False, SourceIngestJob.finished==False).\
        order_by(SourceIngestJob.id).\
        with_for_update().\
        first()
    if job is None:
        return None
    job_id = job.id
    lightcurve_filename = job.lightcurve_filename
    rank = job.process_rank
    size = job.process_size

    job.started = True
    job.slurm_job_id = os.getenv('SLURM_JOB_ID')
    job.datetime = datetime.now()
    db.session.commit()

    return job_id, lightcurve_filename, rank, size


def reset_job(job_id):
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.started = False
    db.session.commit()


def finish_job(job_id):
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.finished = True
    job.datetime = datetime.now()
    db.session.commit()


def write_to_record(msg, mpi_rank):
    now = datetime.now()
    with open('record.txt', 'a') as f:
        f.write(f'{mpi_rank}: {msg} ({now})')


def upload_sources(lightcurve_filename, source_set, mpi_rank):

    write_to_record('Reading from database', mpi_rank)
    sources_db = db.session.query(Source).\
        with_for_update().\
        filter(Source.lightcurve_filename == lightcurve_filename).\
        all()
    write_to_record('-- In database session', mpi_rank)
    keys_db = set([(s.object_id_g, s.object_id_r, s.object_id_i)
                   for s in sources_db])

    for source in source_set:
        key = (source.object_id_g, source.object_id_r, source.object_id_i)
        if key not in keys_db:
            db.session.add(source)
    write_to_record('-- Committing to database', mpi_rank)
    db.session.commit()
    write_to_record('Session Ended', mpi_rank)


def ingest_sources(mpi_rank, nepochs_min=20, shutdown_time=5, single_job=False):
    while True:
        job_enddate = fetch_job_enddate()
        if job_enddate:
            script_enddate = job_enddate - timedelta(minutes=shutdown_time)

        job_data = fetch_job()
        if job_data is None:
            return

        job_id, lightcurve_filename, rank, size = job_data
        print(f'Job {job_id}: Lightcurve file: {lightcurve_filename}')
        print(f'Job {job_id}: Rank: {rank}')
        print(f'Job {job_id}: Size: {size}')

        source_set = set()
        lightcurveFile = LightcurveFile(lightcurve_filename, proc_rank=rank,
                                        proc_size=size, apply_catmask=True)
        for obj in lightcurveFile:
            if obj.lightcurve.nepochs < nepochs_min:
                continue

            obj.locate_siblings()

            source = convert_obj_to_source(obj, lightcurve_filename)
            source_set.add(source)

            if job_enddate and datetime.now() >= script_enddate:
                print(f'Within {shutdown_time} minutes of job end, '
                      f'shutting down...')
                reset_job(job_id)
                time.sleep(2 * 60 * shutdown_time)
                return

        num_sources = len(source_set)
        print(f'Job {job_id}: Uploading {num_sources} sources to database')
        upload_sources(lightcurve_filename, source_set, mpi_rank)
        print(f'Job {job_id}: Upload complete')

        finish_job(job_id)
        print(f'Job {job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.rank
    ingest_sources(mpi_rank)
