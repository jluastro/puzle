#! /usr/bin/env python
"""
identify_stars.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.sql.expression import func
from zort.radec import return_shifted_ra
from scipy.spatial import cKDTree

from puzle.models import Source, SourceIngestJob, Star, StarIngestJob
from puzle.utils import fetch_job_enddate, return_DR5_dir, \
    lightcurve_file_to_field_id, get_logger
from puzle.ulensdb import insert_db_id, remove_db_id, get_logger
from puzle import db

logger = get_logger(__name__)


def fetch_job():
    insert_db_id()  # get permission to make a db connection

    db.session.execute('LOCK TABLE star_ingest_job '
                       'IN ROW EXCLUSIVE MODE;')
    job = db.session.query(StarIngestJob).\
        outerjoin(SourceIngestJob, StarIngestJob.source_ingest_job_id == SourceIngestJob.id).\
        filter(SourceIngestJob.finished == True,
               StarIngestJob.started == False).\
        order_by(func.random()).\
        with_for_update().\
        first()
    if job is None:
        return None
    source_job_id = job.source_ingest_job_id

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        slurm_job_id = os.getenv('SLURM_JOB_ID')
    else:
        rank = 0
        slurm_job_id = 0
    job.slurm_job_rank = rank
    job.started = True
    job.slurm_job_id = slurm_job_id
    job.datetime_started = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection
    return source_job_id


def reset_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarIngestJob).filter(
        StarIngestJob.source_ingest_job_id == source_job_id).one()
    job.started = False
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def finish_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarIngestJob).filter(
        StarIngestJob.source_ingest_job_id == source_job_id).one()
    job.finished = True
    job.datetime_finished = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def _parse_object_int(attr):
    if attr == 'None':
        return None
    else:
        return int(attr)


def csv_line_to_source(line):
    attrs = line.replace('\n', '').split(',')
    source = Source(id=attrs[0],
                    object_id_g=_parse_object_int(attrs[1]),
                    object_id_r=_parse_object_int(attrs[2]),
                    object_id_i=_parse_object_int(attrs[3]),
                    lightcurve_position_g=_parse_object_int(attrs[4]),
                    lightcurve_position_r=_parse_object_int(attrs[5]),
                    lightcurve_position_i=_parse_object_int(attrs[6]),
                    lightcurve_filename=attrs[7],
                    ra=float(attrs[8]),
                    dec=float(attrs[9]),
                    ingest_job_id=int(attrs[10]))
    return source


def fetch_sources(source_job_id):
    DR5_dir = return_DR5_dir()
    dir = '%s/sources_%s' % (DR5_dir, str(source_job_id)[:3])

    if not os.path.exists(dir):
        logger.error('Source directory missing!')
        return

    fname = f'{dir}/sources.{source_job_id:06}.txt'
    lines = open(fname, 'r').readlines()[1:]

    sources = []
    for line in lines:
        source = csv_line_to_source(line)
        sources.append(source)

    return sources


def _export_stars(source_job_id, stars):
    DR5_dir = return_DR5_dir()
    dir = '%s/stars_%s' % (DR5_dir, str(source_job_id)[:3])

    if not os.path.exists(dir):
        os.makedirs(dir)

    fname = f'{dir}/stars.{source_job_id:06}.txt'
    if os.path.exists(fname):
        os.remove(fname)

    with open(fname, 'w') as f:
        header = 'id,'
        header += 'ra,'
        header += 'dec,'
        header += 'ingest_job_id,'
        header += 'source_ids'
        f.write(f'{header}\n')

        for i, star in enumerate(stars):
            star.id = '%s_%i' % (source_job_id, i)
            star_line = star_to_csv_line(star)
            f.write(f'{star_line}\n')


def export_stars(source_job_id, sources):
    if len(sources) == 0:
        _export_stars(source_job_id, [])
        return

    radec = []
    source_ids = []
    for source in sources:
        field_id = lightcurve_file_to_field_id(source.lightcurve_filename)
        ra, dec = source.ra, source.dec
        ra_shifted = return_shifted_ra(ra, field_id)
        radec.append((ra_shifted, dec))
        source_ids.append(source.id)

    kdtree = cKDTree(np.array(radec))
    radius_deg = 2 / 3600.
    star_keys = set()
    stars = []

    for ((ra, dec), source_id) in zip(radec, source_ids):
        idx_arr = kdtree.query_ball_point((ra, dec), radius_deg)
        ids = [source_ids[idx] for idx in idx_arr]
        ids.sort()
        key = tuple(ids)
        if key not in star_keys:
            star_keys.add(key)

            if ra < 0:
                ra += 360
            elif ra > 360:
                ra -= 360
            star = Star(source_ids=ids,
                        ra=ra, dec=dec,
                        ingest_job_id=source_job_id)
            stars.append(star)

    _export_stars(source_job_id, stars)


def star_to_csv_line(star):
    line = '%s,' % star.id
    line += '%s,' % star.ra
    line += '%s,' % star.dec
    line += '%s,' % star.ingest_job_id
    ids = [str(source_id) for source_id in star.source_ids]
    line += '"{%s}"' % ','.join(ids)
    return line


def identify_stars(source_job_id):
    logger.info(f'Job {source_job_id}: Fetching sources')
    sources = fetch_sources(source_job_id)

    num_sources = len(sources)
    logger.info(f'Job {source_job_id}: Exporting {num_sources} stars to disk')
    export_stars(source_job_id, sources)
    logger.info(f'Job {source_job_id}: Export complete')

    finish_job(source_job_id)
    logger.info(f'Job {source_job_id}: Job complete')


def identify_stars_script(shutdown_time=10, single_job=False):
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

        identify_stars(source_job_id)

        if single_job:
            return


if __name__ == '__main__':
    identify_stars_script()
