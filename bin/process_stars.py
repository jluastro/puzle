#! /usr/bin/env python
"""
process_stars.py
"""
import time
import os
from datetime import datetime, timedelta
from sqlalchemy.sql.expression import func
import logging
from collections import defaultdict

from puzle.models import Source, StarIngestJob, Star, StarProcessJob, Candidate
from puzle.utils import fetch_job_enddate, return_DR3_dir
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.stats import calculate_eta, fit_event, \
    calculate_eta_on_residuals, \
    return_eta_threshold, RF_THRESHOLD
from puzle import catalog
from puzle import db

logger = logging.getLogger(__name__)


def fetch_job():
    insert_db_id()  # get permission to make a db connection

    db.session.execute('LOCK TABLE star_process_job '
                       'IN ROW EXCLUSIVE MODE;')
    job = db.session.query(StarProcessJob).\
        outerjoin(StarIngestJob,
                  StarProcessJob.source_ingest_job_id == StarIngestJob.source_ingest_job_id).\
        filter(StarIngestJob.finished == True,
               StarProcessJob.started == False).\
        order_by(func.random()).\
        with_for_update().\
        first()
    if job is None:
        return None
    source_job_id = job.source_ingest_job_id

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    job.slurm_job_rank = rank
    job.started = True
    job.slurm_job_id = os.getenv('SLURM_JOB_ID')
    job.datetime_started = datetime.now()
    db.session.commit()

    remove_db_id()  # release permission for this db connection
    return source_job_id


def reset_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarProcessJob).filter(
        StarProcessJob.source_ingest_job_id == source_job_id).one()
    job.started = False
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def finish_job(source_job_id, job_stats):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarProcessJob).filter(
        StarProcessJob.source_ingest_job_id == source_job_id).one()
    job.finished = True
    job.datetime_finished = datetime.now()
    job.num_stars = job_stats['num_stars']
    job.num_sources = job_stats['num_sources']
    job.num_objs = job_stats['num_objs']
    job.num_objs_pass_eta = job_stats['num_objs_pass_eta']
    job.num_stars_pass_eta = job_stats['num_stars_pass_eta']
    job.num_objs_pass_rf = job_stats['num_objs_pass_rf']
    job.num_stars_pass_rf = job_stats['num_stars_pass_rf']
    job.num_objs_pass_eta_residual = job_stats['num_objs_pass_eta_residual']
    job.num_stars_pass_eta_residual = job_stats['num_stars_pass_eta_residual']
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def csv_line_to_star_and_sources(line):
    star_id = float(line.split(',')[0])
    ra = float(line.split(',')[1])
    dec = float(line.split(',')[2])
    ingest_job_id = int(line.split(',')[3])
    source_ids = line.split('{')[1].split('}')[0].split(',')

    star = Star(id=star_id, source_ids=source_ids,
                ra=ra, dec=dec,
                ingest_job_id=ingest_job_id)
    return star


def fetch_stars_and_sources(source_job_id):
    DR3_dir = return_DR3_dir()
    dir = '%s/stars_%s' % (DR3_dir, str(source_job_id)[:3])

    if not os.path.exists(dir):
        logging.error('Source directory missing!')
        return

    fname = f'{dir}/stars.{source_job_id:06}.txt'
    lines = open(fname, 'r').readlines()[1:]

    source_ids = []
    source_to_star_dict = {}
    for i, line in enumerate(lines):
        star = csv_line_to_star_and_sources(line)
        source_ids.extend(star.source_ids)
        for source_id in star.source_ids:
            source_to_star_dict[source_id] = star

    insert_db_id()
    sources_db = db.session.query(Source).filter(Source.id.in_(source_ids)).all()
    remove_db_id()
    star_to_source_dict = defaultdict(list)
    for source_db in sources_db:
        star = source_to_star_dict[source_db.id]
        star_to_source_dict[star].append(source_db)

    return list(star_to_source_dict.items())


def filter_stars_to_candidates(source_job_id, stars_and_sources):
    logger.info(f'Job {source_job_id}: Calculating eta')
    num_stars = 0
    num_sources = 0
    num_objs = 0
    eta_idxs = []
    for i, (star, sources) in enumerate(stars_and_sources):
        num_stars += 1
        for j, source in enumerate(sources):
            num_sources += 1
            for k, obj in enumerate(source.zort_source.objects):
                num_objs += 1
                eta = calculate_eta(obj.lightcurve.mag)
                eta_threshold = return_eta_threshold(obj.nepochs)
                if eta <= eta_threshold:
                    eta_idxs.append((i, j, k))

    logger.info(f'Job {source_job_id}: '
                f'{num_stars} Stars | '
                f'{num_sources} Sources | '
                f'{num_objs} Objects')
    num_objs_pass_eta = len(eta_idxs)
    num_stars_pass_eta = len(set([idx[0] for idx in eta_idxs]))

    logger.info(f'Job {source_job_id}: '
                f'{num_stars_pass_eta} stars pass eta cut | '
                f'{num_objs_pass_eta} objects pass eta cut')

    insert_db_id()
    ulens_con = catalog.ulens_con()
    rf_idxs = []
    for (i, j, k) in eta_idxs:
        _, sources = stars_and_sources[i]
        source = sources[j]
        obj = source.zort_source.objects[k]
        rf_score = catalog.query_ps1_psc(obj.ra, obj.dec,
                                         con=ulens_con)
        if rf_score is None or rf_score.rf_score >= RF_THRESHOLD:
            rf_idxs.append((i, j, k))
    ulens_con.close()
    remove_db_id()

    num_objs_pass_rf = len(rf_idxs)
    rf_idxs = list(set(rf_idxs))
    num_stars_pass_rf = len(set([idx[0] for idx in rf_idxs]))

    logger.info(f'Job {source_job_id}: '
                f'{num_stars_pass_rf} stars pass rf_score cut | '
                f'{num_objs_pass_rf} objects pass rf_score cut')

    eta_residual_idxs = []
    for (i, j, k) in rf_idxs:
        _, sources = stars_and_sources[i]
        source = sources[j]
        obj = source.zort_source.objects[k]
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.magerr
        magerr = obj.lightcurve.magerr
        eta_residual = calculate_eta_on_residuals(hmjd, mag, magerr)
        eta_threshold = return_eta_threshold(obj.nepochs)
        if eta_residual is not None and eta_residual > eta_threshold:
            eta_residual_idxs.append((i, j, k))

    num_objs_pass_eta_residual = len(eta_residual_idxs)
    eta_residual_idxs = list(set(eta_residual_idxs))
    num_stars_pass_eta_residual = len(set([idx[0] for idx in eta_residual_idxs]))

    logger.info(f'Job {source_job_id}: '
                f'{num_stars_pass_eta_residual} source pass eta_residual cut | '
                f'{num_objs_pass_eta_residual} objects pass eta_residual cut')

    logger.info(f'Job {source_job_id}: Assembling candidates')
    insert_db_id()
    ulens_con = catalog.ulens_con()
    candidates = []
    unique_star_idxs = list(set([i for i, _, _ in eta_residual_idxs]))
    for star_idx in unique_star_idxs:
        star, sources = stars_and_sources[star_idx]
        source_obj_idxs = [(j, k) for i, j, k in eta_residual_idxs if i == star_idx]
        source_id_arr = []
        filter_id_arr = []
        eta_arr = []
        rf_arr = []
        eta_residual_arr = []
        eta_threshold_arr = []
        for j, k in source_obj_idxs:
            source = sources[j]
            obj = source.zort_source.objects[k]
            eta = calculate_eta(obj.lightcurve.mag)
            eta_threshold = return_eta_threshold(obj.nepochs)
            rf_score = catalog.query_ps1_psc(obj.ra, obj.dec,
                                             con=ulens_con)
            hmjd = obj.lightcurve.hmjd
            mag = obj.lightcurve.magerr
            magerr = obj.lightcurve.magerr
            eta_residual = calculate_eta_on_residuals(hmjd, mag, magerr)

            source_id_arr.append(source.id)
            filter_id_arr.append(obj.filterid)
            eta_arr.append(eta)
            rf_arr.append(rf_score)
            eta_residual_arr.append(eta_residual)
            eta_threshold_arr.append(eta_threshold)

        cand = Candidate(id=star.id,
                         source_ids=source_id_arr,
                         filter_ids=filter_id_arr,
                         ra=star.ra,
                         dec=star.dec,
                         ingest_job_id=star.ingest_job_id,
                         etas=eta_arr,
                         rf_scores=rf_arr,
                         eta_residuals=eta_residual_arr,
                         eta_thresholds=eta_threshold_arr)
        candidates.append(cand)

    ulens_con.close()
    remove_db_id()

    job_stats = {'num_stars': num_stars,
                 'num_sources': num_sources,
                 'num_objs': num_objs,
                 'num_objs_pass_eta': num_objs_pass_eta,
                 'num_stars_pass_eta': num_stars_pass_eta,
                 'num_objs_pass_rf': num_objs_pass_rf,
                 'num_stars_pass_rf': num_stars_pass_rf,
                 'num_objs_pass_eta_residual': num_objs_pass_eta_residual,
                 'num_stars_pass_eta_residual': num_stars_pass_eta_residual}
    return candidates, job_stats


def upload_candidates(candidates):
    insert_db_id()  # get permission to make a db connection
    for cand in candidates:
        db.session.add(cand)
    db.session.commit()
    remove_db_id()  # release permission for this db connection


def process_stars(shutdown_time=10, single_job=False):
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

        stars_and_sources = fetch_stars_and_sources(source_job_id)

        num_stars = len(stars_and_sources)
        logger.info(f'Job {source_job_id}: Processing {num_stars} stars')
        candidates, job_stats = filter_stars_to_candidates(source_job_id,
                                                           stars_and_sources)
        num_candidates = len(candidates)
        logger.info(f'Job {source_job_id}: Uploading {num_candidates} candidates')
        upload_candidates(candidates)
        logger.info(f'Job {source_job_id}: Processing complete')

        finish_job(source_job_id, job_stats)
        logger.info(f'Job {source_job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_stars()
