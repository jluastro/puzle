#! /usr/bin/env python
"""
identify_sources.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from zort.lightcurveFile import LightcurveFile
from sqlalchemy.sql.expression import func
import logging
import pickle

from puzle.models import Source, SourceIngestJob
from puzle.utils import fetch_job_enddate, return_DR4_dir, fetch_lightcurve_rcids
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle import db

logger = logging.getLogger(__name__)


def object_in_bounds(obj, ra_start, ra_end, dec_start, dec_end):
    if (obj.ra >= ra_start) and (obj.ra < ra_end) \
            and (obj.dec >= dec_start) and (obj.dec < dec_end):
        return True
    else:
        return False


def convert_obj_to_source(obj, lightcurve_filename, job_id):

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
        'lightcurve_filename': lightcurve_filename,
        'ingest_job_id': job_id
    }

    obj_filt = filter_dict[obj.filterid]
    source_dict[f'object_id_{obj_filt}'] = obj.object_id
    source_dict[f'lightcurve_position_{obj_filt}'] = obj.lightcurve_position

    ra_arr = [obj.ra]
    dec_arr = [obj.dec]

    if obj.siblings:
        for sib in obj.siblings:
            sib_filt = filter_dict[sib.filterid]
            source_dict[f'object_id_{sib_filt}'] = sib.object_id
            source_dict[f'lightcurve_position_{sib_filt}'] = sib.lightcurve_position
            ra_arr.append(sib.ra)
            dec_arr.append(sib.dec)

    source_dict['ra'] = np.mean(ra_arr)
    source_dict['dec'] = np.mean(dec_arr)

    return Source(**source_dict)


def fetch_job():
    insert_db_id()  # get permission to make a db connection

    db.session.execute('LOCK TABLE source_ingest_job '
                       'IN ROW EXCLUSIVE MODE;')
    job = db.session.query(SourceIngestJob).\
        filter(SourceIngestJob.started == False).\
        order_by(func.random()).\
        with_for_update().\
        first()
    if job is None:
        return None
    job_id = job.id
    ra_start = job.ra_start
    ra_end = job.ra_end
    dec_start = job.dec_start
    dec_end = job.dec_end

    if 'SLURMD_NODENAME' in os.environ:
        slurm_job_id = os.getenv('SLURM_JOB_ID')
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        slurm_job_id = 0
        rank = 0
    job.slurm_job_rank = rank
    job.started = True
    job.slurm_job_id = slurm_job_id
    job.datetime_started = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection
    return job_id, ra_start, ra_end, dec_start, dec_end


def reset_job(job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.started = False
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def finish_job(job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(SourceIngestJob).filter(
        SourceIngestJob.id == job_id).one()
    job.finished = True
    job.datetime_finished = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection


def source_to_csv_line(source, source_id):
    line = '%s_%s,' % (source.ingest_job_id, source_id)
    line += '%s,' % str(source.object_id_g)
    line += '%s,' % str(source.object_id_r)
    line += '%s,' % str(source.object_id_i)
    line += '%s,' % str(source.lightcurve_position_g)
    line += '%s,' % str(source.lightcurve_position_r)
    line += '%s,' % str(source.lightcurve_position_i)
    line += '%s,' % source.lightcurve_filename
    line += '%s,' % source.ra
    line += '%s,' % source.dec
    line += '%s' % source.ingest_job_id
    return line


def export_sources(job_id, source_list):
    DR4_dir = return_DR4_dir()
    dir = '%s/sources_%s' % (DR4_dir, str(job_id)[:3])

    if not os.path.exists(dir):
        os.makedirs(dir)

    source_exported = []
    fname = f'{dir}/sources.{job_id:06}.txt'
    if os.path.exists(fname):
        os.remove(fname)

    sources_map = {}
    with open(fname, 'w') as f:
        header = 'id,'
        header += 'object_id_g,'
        header += 'object_id_r,'
        header += 'object_id_i,'
        header += 'lightcurve_position_g,'
        header += 'lightcurve_position_r,'
        header += 'lightcurve_position_i,'
        header += 'lightcurve_filename,'
        header += 'ra,'
        header += 'dec,'
        header += 'ingest_job_id'
        f.write(f'{header}\n')

        source_keys = set()
        source_id = 0
        for source in source_list:
            key = (source.object_id_g, source.object_id_r, source.object_id_i)
            if key not in source_keys:
                source_keys.add(key)
                source_line = source_to_csv_line(source, source_id)
                source_exported.append(source)
                line = f'{source_line}\n'
                f.write(line)

                sources_map_key = '%s_%s' % (source.ingest_job_id, source_id)
                sources_map[sources_map_key] = f.tell() - len(line)
                source_id += 1

    map_filename = fname.replace('.txt', '.sources_map')
    with open(map_filename, 'wb') as fileObj:
        pickle.dump(sources_map, fileObj)


def identify_sources(job_data, nepochs_min=20):
    job_id, ra_start, ra_end, dec_start, dec_end = job_data
    logger.info(f'Job {job_id}: ra: {ra_start:.5f} to {ra_end:.5f} ')
    logger.info(f'Job {job_id}: dec: {dec_start:.5f} to {dec_end:.5f} ')

    lightcurve_rcids = fetch_lightcurve_rcids(ra_start, ra_end, dec_start, dec_end)

    source_list = []
    for lightcurve_file, rcids_to_read in lightcurve_rcids:
        lightcurveFile = LightcurveFile(lightcurve_file, apply_catmask=True,
                                        rcids_to_read=rcids_to_read)

        for obj in lightcurveFile:
            if obj.lightcurve.nepochs < nepochs_min:
                continue

            if not object_in_bounds(obj, ra_start, ra_end, dec_start, dec_end):
                continue

            obj.locate_siblings()

            source = convert_obj_to_source(obj, lightcurve_file, job_id)
            source_list.append(source)

    num_sources = len(source_list)
    logger.info(f'Job {job_id}: Exporting {num_sources} sources to disk')
    export_sources(job_id, source_list)
    logger.info(f'Job {job_id}: Export complete')

    finish_job(job_id)
    logger.info(f'Job {job_id}: Job complete')


def identify_sources_script(nepochs_min=20, shutdown_time=10, single_job=False):
    job_enddate = fetch_job_enddate()
    if job_enddate:
        script_enddate = job_enddate - timedelta(minutes=shutdown_time)
        logger.info('Script End Date: %s' % script_enddate)

    while True:
        job_data = fetch_job()
        if job_data is None:
            return

        if job_enddate and datetime.now() >= script_enddate:
            logger.info(f'Within {shutdown_time} minutes of job end, '
                        f'shutting down...')
            reset_job(job_data[0])
            time.sleep(2 * 60 * shutdown_time)
            return

        identify_sources(job_data, nepochs_min=nepochs_min)

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    identify_sources_script()
