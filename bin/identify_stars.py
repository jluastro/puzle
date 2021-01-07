#! /usr/bin/env python
"""
identify_stars.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from zort.radec import lightcurve_file_is_pole
from sqlalchemy.sql.expression import func
import logging
from scipy.spatial import cKDTree

from puzle.models import Source, SourceIngestJob, Star, StarIngestJob
from puzle.utils import fetch_job_enddate
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle import db

logger = logging.getLogger(__name__)


def fetch_job():
    insert_db_id()  # get permission to make a db connection

    db.session.execute('LOCK TABLE star_ingest_job '
                       'IN ROW EXCLUSIVE MODE;')
    job = db.session.query(StarIngestJob).\
        outerjoin(SourceIngestJob, StarIngestJob.source_ingest_job_id == SourceIngestJob.id).\
        filter(SourceIngestJob.finished == True,
               StarIngestJob.started == False,
               StarIngestJob.finished == False).\
        order_by(func.random()).\
        with_for_update().\
        first()
    if job is None:
        return None
    source_job_id = job.source_ingest_job_id

    job.started = True
    job.slurm_job_id = os.getenv('SLURM_JOB_ID')
    job.datetime_started = datetime.now()
    db.session.commit()

    remove_db_id()  # release permission for this db connection
    return source_job_id


def reset_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarIngestJob).filter(
        StarIngestJob.source_ingest_job_id == source_job_id).one()
    job.started = False
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def finish_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarIngestJob).filter(
        StarIngestJob.source_ingest_job_id == source_job_id).one()
    job.finished = True
    job.datetime_finished = datetime.now()
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def fetch_sources(source_job_id):
    insert_db_id()  # get permission to make a db connection
    sources = db.session.query(Source).filter(
        Source.ingest_job_id == source_job_id).all()
    remove_db_id()  # release permission for this db connection
    return sources


def export_stars(source_job_id, sources):
    radec = []
    source_ids = []
    for source in sources:
        is_pole = lightcurve_file_is_pole(source.lightcurve_filename)
        ra, dec = source.ra, source.dec
        if is_pole and ra > 180:
            ra -= 360

        radec.append((ra, dec))
        source_ids.append(source.id)

    kdtree = cKDTree(np.array(radec))
    radius_deg = 2 / 3600.
    star_keys = set()

    fname = f'stars.{source_job_id:06}.txt'
    with open(fname, 'w') as f:
        header = 'ra,'
        header += 'dec,'
        header += 'ingest_job_id,'
        header += 'ids'
        f.write(f'{header}\n')

        for ((ra, dec), source_id) in zip(radec, source_ids):
            idx_arr = kdtree.query_ball_point((ra, dec), radius_deg)
            ids = [source_ids[idx] for idx in idx_arr]
            ids.sort()
            key = tuple(ids)
            if key not in star_keys:
                star_keys.add(key)
                star = Star(source_ids=ids,
                            ra=ra, dec=dec,
                            ingest_job_id=source_job_id)
                star_line = star_to_csv_line(star)
                f.write(f'{star_line}\n')


def star_to_csv_line(star):
    line = '%s,' % star.ra
    line += '%s,' % star.dec
    line += '%s,' % star.ingest_job_id
    ids = [str(source_id) for source_id in star.source_ids]
    line += '{%s}' % ','.join(ids)
    return line


def identify_stars(shutdown_time=10, single_job=False):
    job_enddate = fetch_job_enddate()
    if job_enddate:
        script_enddate = job_enddate - timedelta(minutes=shutdown_time)
        logger.info('Script End Date: %s' % script_enddate)

    while True:

        source_job_id = fetch_job()
        if source_job_id is None:
            return

        if job_enddate and datetime.now() >= script_enddate:
            logger.info(f'Within {shutdown_time} minutes of job end, '
                        f'shutting down...')
            reset_job(source_job_id)
            time.sleep(2 * 60 * shutdown_time)
            return

        sources = fetch_sources(source_job_id)
        n_sources = len(sources)
        logger.info(f'Job {source_job_id}: {n_sources} sources')

        export_stars(source_job_id, sources)
        logger.info(f'Job {source_job_id}: Export complete')

        finish_job(source_job_id)
        logger.info(f'Job {source_job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    identify_stars()
