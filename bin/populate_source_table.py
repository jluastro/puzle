#! /usr/bin/env python
"""
populate_source_table.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from zort.lightcurveFile import LightcurveFile
import logging

from puzle.models import Source, SourceIngestJob
from puzle.utils import fetch_job_enddate
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle import db

logger = logging.getLogger(__name__)


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
    insert_db_id()  # get permission to make a db connection

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
    job.datetime_started = datetime.now()
    db.session.commit()

    remove_db_id()  # release permission for this db connection
    return job_id, lightcurve_filename, rank, size


def reset_job(job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.started = False
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def finish_job(job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.finished = True
    job.datetime_finished = datetime.now()
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def upload_sources(lightcurve_filename, source_list):
    insert_db_id()  # get permission to make a db connection
    sources_db = db.session.query(Source).\
        with_for_update().\
        filter(Source.lightcurve_filename == lightcurve_filename).\
        all()
    keys_db = set([(s.object_id_g, s.object_id_r, s.object_id_i)
                   for s in sources_db])

    for source in source_list:
        key = (source.object_id_g, source.object_id_r, source.object_id_i)
        if key not in keys_db:
            db.session.add(source)
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def ingest_sources(nepochs_min=20, shutdown_time=5, single_job=False):
    job_enddate = fetch_job_enddate()
    if job_enddate:
        script_enddate = job_enddate - timedelta(minutes=shutdown_time)
        logger.info('Script End Date: %s' % script_enddate)

    while True:
        job_data = fetch_job()
        if job_data is None:
            return

        job_id, lightcurve_filename, rank, size = job_data
        logger.info(f'Job {job_id}: Lightcurve file: {lightcurve_filename}')
        logger.info(f'Job {job_id}: Rank: {rank}')
        logger.info(f'Job {job_id}: Size: {size}')

        source_list = []
        lightcurveFile = LightcurveFile(lightcurve_filename, proc_rank=rank,
                                        proc_size=size, apply_catmask=True)
        for obj in lightcurveFile:
            if obj.lightcurve.nepochs < nepochs_min:
                continue

            obj.locate_siblings()

            source = convert_obj_to_source(obj, lightcurve_filename)
            source_list.append(source)

            if job_enddate and datetime.now() >= script_enddate:
                logger.info(f'Within {shutdown_time} minutes of job end, '
                            f'shutting down...')
                reset_job(job_id)
                time.sleep(2 * 60 * shutdown_time)
                return

        num_sources = len(source_list)
        logger.info(f'Job {job_id}: Uploading {num_sources} sources to database')
        upload_sources(lightcurve_filename, source_list)
        logger.info(f'Job {job_id}: Upload complete')

        finish_job(job_id)
        logger.info(f'Job {job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ingest_sources()
