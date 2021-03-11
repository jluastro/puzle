#! /usr/bin/env python
"""
process_stars.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from puzle.models import Source, StarIngestJob, Star, StarProcessJob, Candidate
from puzle.utils import fetch_job_enddate, return_DR3_dir
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.stats import calculate_eta, \
    calculate_eta_on_residuals, RF_THRESHOLD
from puzle.fit import fit_event
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
        order_by(StarProcessJob.priority.asc()).\
        with_for_update().\
        first()
    if job is None:
        return None
    source_job_id = job.source_ingest_job_id

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        rank = 0
    job.slurm_job_rank = rank
    job.started = True
    job.slurm_job_id = os.getenv('SLURM_JOB_ID')
    job.datetime_started = datetime.now()
    db.session.commit()
    db.session.close()

    remove_db_id()  # release permission for this db connection
    return source_job_id


def reset_job(source_job_id):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarProcessJob).filter(
        StarProcessJob.source_ingest_job_id == source_job_id).one()
    job.started = False
    db.session.commit()
    db.session.close()
    remove_db_id()  # release permission for this db connection


def finish_job(source_job_id, job_stats):
    insert_db_id()  # get permission to make a db connection
    job = db.session.query(StarProcessJob).filter(
        StarProcessJob.source_ingest_job_id == source_job_id).one()
    job.finished = True
    job.uploaded = True
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
    db.session.close()
    remove_db_id()  # release permission for this db connection


def csv_line_to_star_and_sources(line):
    star_id = str(line.split(',')[0])
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
    db.session.close()
    remove_db_id()
    star_to_source_dict = defaultdict(list)
    for source_db in sources_db:
        star = source_to_star_dict[source_db.id]
        star_to_source_dict[star].append(source_db)

    return list(star_to_source_dict.items())


def filter_stars_to_candidates(source_job_id, stars_and_sources,
                               nepochs_min=20):
    logger.info(f'Job {source_job_id}: Calculating eta')
    num_stars = 0
    num_sources = 0
    num_objs = 0
    idxs = []
    eta_arr = []
    for i, (star, sources) in enumerate(stars_and_sources):
        num_stars += 1
        for j, source in enumerate(sources):
            num_sources += 1
            for k, obj in enumerate(source.zort_source.objects):
                if obj.nepochs < nepochs_min:
                    continue
                num_objs += 1
                eta = calculate_eta(obj.lightcurve.mag)
                idxs.append((i, j, k))
                eta_arr.append(eta)
    eta_arr = np.array(eta_arr)
    idxs = np.array(idxs)
    eta_threshold = np.percentile(eta_arr, 1)
    eta_cond = np.where(eta_arr <= eta_threshold)[0]
    eta_idxs = idxs[eta_cond]

    logger.info(f'Job {source_job_id}: '
                f'{num_stars} Stars | '
                f'{num_sources} Sources | '
                f'{num_objs} Objects')
    num_objs_pass_eta = len(eta_idxs)
    num_stars_pass_eta = len(set([idx[0] for idx in eta_idxs]))

    logger.info(f'Job {source_job_id}: '
                f'{num_stars_pass_eta} stars pass eta cut | '
                f'{num_objs_pass_eta} objects pass eta cut | '
                f'eta_threshold = {eta_threshold:.3f}')

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

        n_days_arr = []
        source_id_arr = []
        filter_id_arr = []
        for idx, (j, k) in enumerate(source_obj_idxs):
            source = sources[j]
            obj = source.zort_source.objects[k]
            n_days = len(np.unique(np.floor(obj.lightcurve.hmjd)))

            source_id_arr.append(source.id)
            filter_id_arr.append(obj.filterid)
            n_days_arr.append(n_days)

        idx_best = int(np.argmax(n_days_arr))
        j, k = source_obj_idxs[idx_best]
        source = sources[j]
        obj = source.zort_source.objects[k]

        eta = calculate_eta(obj.lightcurve.mag)
        rf_score_obj = catalog.query_ps1_psc(obj.ra, obj.dec,
                                             con=ulens_con)
        if rf_score_obj is None:
            rf_score = None
        else:
            rf_score = rf_score_obj.rf_score
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.magerr
        magerr = obj.lightcurve.magerr
        eta_residual = calculate_eta_on_residuals(hmjd, mag, magerr)

        fit_data = fit_event(hmjd, mag, magerr)
        t_0, t_E, f_0, f_1, chi_squared_delta, chi_squared_flat, a_type = fit_data

        cand = Candidate(id=star.id,
                         source_id_arr=source_id_arr,
                         filter_id_arr=filter_id_arr,
                         ra=star.ra,
                         dec=star.dec,
                         ingest_job_id=star.ingest_job_id,
                         eta_best=eta,
                         rf_score_best=rf_score,
                         eta_residual_best=eta_residual,
                         eta_threshold=eta_threshold,
                         t_E_best=t_E,
                         t_0_best=t_0,
                         f_0_best=f_0,
                         f_1_best=f_1,
                         a_type_best=a_type,
                         chi_squared_flat_best=chi_squared_flat,
                         chi_squared_delta_best=chi_squared_delta,
                         idx_best=idx_best)
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
    db.session.close()
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
        if num_stars > 0:
            logger.info(f'Job {source_job_id}: Processing {num_stars} stars')
            candidates, job_stats = filter_stars_to_candidates(source_job_id,
                                                               stars_and_sources)
            num_candidates = len(candidates)
            logger.info(f'Job {source_job_id}: Uploading {num_candidates} candidates')
            upload_candidates(candidates)
            logger.info(f'Job {source_job_id}: Processing complete')
        else:
            logger.info(f'Job {source_job_id}: No stars, skipping process')
            job_stats = {'num_stars': 0,
                         'num_sources': 0,
                         'num_objs': 0,
                         'num_objs_pass_eta': 0,
                         'num_stars_pass_eta': 0,
                         'num_objs_pass_rf': 0,
                         'num_stars_pass_rf': 0,
                         'num_objs_pass_eta_residual': 0,
                         'num_stars_pass_eta_residual': 0}

        finish_job(source_job_id, job_stats)
        logger.info(f'Job {source_job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_stars()
