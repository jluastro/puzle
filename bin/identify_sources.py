#! /usr/bin/env python
"""
identify_sources.py
"""
import time
import os
import glob
import numpy as np
from datetime import datetime, timedelta
from zort.lightcurveFile import LightcurveFile
from zort.radec import return_shifted_ra, return_ZTF_RCID_corners
from sqlalchemy.sql.expression import func
from shapely.geometry.polygon import Polygon
import logging

from puzle.models import Source, SourceIngestJob
from puzle.utils import fetch_job_enddate, lightcurve_file_to_ra_dec
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle import db

logger = logging.getLogger(__name__)


def fetch_lightcurve_rcids(ra_start, ra_end, dec_start, dec_end):
    lightcurve_files = glob.glob('field*txt')
    lightcurve_files.sort()

    lightcurve_rcids_arr = []
    for i, lightcurve_file in enumerate(lightcurve_files):
        field_id = int(lightcurve_file.split('_')[0].replace('field', ''))

        ra0, ra1, dec0, dec1 = lightcurve_file_to_ra_dec(lightcurve_file)
        if ra1 < ra0:
            ra0_shifted = return_shifted_ra(ra0, field_id)
            ra1_shifted = return_shifted_ra(ra1, field_id)
        else:
            ra0_shifted = ra0
            ra1_shifted = ra1

        file_polygon = Polygon([(ra0_shifted, dec0),
                                (ra0_shifted, dec1),
                                (ra1_shifted, dec1),
                                (ra1_shifted, dec0)])

        if ra0_shifted < ra0 and ra_start > 180:
            ra_start_shifted = ra_start - 360
            ra_end_shifted = ra_end - 360
        elif ra1_shifted > ra1 and ra_end < 180:
            ra_start_shifted = ra_start + 360
            ra_end_shifted = ra_end + 360
        else:
            ra_start_shifted = ra_start
            ra_end_shifted = ra_end

        job_polygon = Polygon([(ra_start_shifted, dec_start),
                               (ra_start_shifted, dec_end),
                               (ra_end_shifted, dec_end),
                               (ra_end_shifted, dec_start)])

        if not file_polygon.intersects(job_polygon):
            continue

        ZTF_RCID_corners = return_ZTF_RCID_corners(field_id)

        rcids_to_read = []
        for rcid, corners in ZTF_RCID_corners.items():
            rcid_polygon = Polygon(corners)
            if rcid_polygon.intersects(job_polygon):
                rcids_to_read.append(rcid)

        if len(rcids_to_read) > 0:
            lightcurve_rcids_arr.append((lightcurve_file, rcids_to_read))

    return lightcurve_rcids_arr


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

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    job.slurm_job_rank = rank
    job.started = True
    job.slurm_job_id = os.getenv('SLURM_JOB_ID')
    job.datetime_started = datetime.now()
    db.session.commit()

    remove_db_id()  # release permission for this db connection
    return job_id, ra_start, ra_end, dec_start, dec_end


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

    dir = 'sources_%s' % str(job_id)[:3]

    if not os.path.exists(dir):
        os.makedirs(dir)

    source_exported = []
    fname = f'{dir}/sources.{job_id:06}.txt'
    if os.path.exists(fname):
        os.remove(fname)

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
                source_id += 1
                f.write(f'{source_line}\n')


def identify_sources(nepochs_min=20, shutdown_time=10, single_job=False):
    job_enddate = fetch_job_enddate()
    if job_enddate:
        script_enddate = job_enddate - timedelta(minutes=shutdown_time)
        logger.info('Script End Date: %s' % script_enddate)

    while True:
        job_data = fetch_job()
        if job_data is None:
            return

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

                if job_enddate and datetime.now() >= script_enddate:
                    logger.info(f'Within {shutdown_time} minutes of job end, '
                                f'shutting down...')
                    reset_job(job_id)
                    time.sleep(2 * 60 * shutdown_time)
                    return

        num_sources = len(source_list)
        logger.info(f'Job {job_id}: Exporting {num_sources} sources to disk')
        export_sources(job_id, source_list)
        logger.info(f'Job {job_id}: Export complete')

        finish_job(job_id)
        logger.info(f'Job {job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    identify_sources()
