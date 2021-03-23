#! /usr/bin/env python
"""
process_stars.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import binned_statistic
import logging
from collections import defaultdict, namedtuple

from puzle.models import Source, StarIngestJob, Star, StarProcessJob, Candidate
from puzle.utils import fetch_job_enddate, return_DR4_dir
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.stats import calculate_eta_on_daily_avg, \
    RF_THRESHOLD, calculate_eta_on_daily_avg_residuals
from puzle import catalog
from puzle import db

logger = logging.getLogger(__name__)

ObjectData = namedtuple('ObjectData', 'eta eta_residual rf_score fit_data')


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
    DR4_dir = return_DR4_dir()
    dir = '%s/stars_%s' % (DR4_dir, str(source_job_id)[:3])

    if not os.path.exists(dir):
        logging.error('Source directory missing!')
        return

    fname = f'{dir}/stars.{source_job_id:06}.txt'
    if not os.path.exists(fname):
        logging.error('Source file missing!')
        return

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


def construct_eta_dct(stars_and_sources, job_stats, obj_data, n_days_min=20):
    num_stars = 0
    num_sources = 0
    num_objs = 0
    num_objs_pass_n_days = 0
    idxs_dct = defaultdict(list)
    eta_dct = defaultdict(list)
    n_epochs_dct = defaultdict(list)
    for i, (star, sources) in enumerate(stars_and_sources):
        num_stars += 1
        for j, source in enumerate(sources):
            num_sources += 1
            for k, obj in enumerate(source.zort_source.objects):
                num_objs += 1
                n_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
                if n_days < n_days_min:
                    continue
                num_objs_pass_n_days += 1
                eta = calculate_eta_on_daily_avg(obj.lightcurve.hmjd,
                                                 obj.lightcurve.mag)

                key = '%i_%i' % (obj.fieldid, obj.filterid)
                idxs_dct[key].append((i, j, k))
                eta_dct[key].append(eta)
                n_epochs_dct[key].append(obj.nepochs)

                obj_key = (i, j, k)
                objectData = ObjectData(eta=eta, eta_residual=None,
                                        rf_score=None, fit_data=None)
                obj_data[obj_key] = objectData

    job_stats['num_stars'] = num_stars
    job_stats['num_sources'] = num_sources
    job_stats['num_objs'] = num_objs
    job_stats['num_objs_pass_n_days'] = num_objs_pass_n_days

    return eta_dct, idxs_dct, n_epochs_dct


def construct_eta_idxs_dct(eta_dct, idxs_dct, n_epochs_dct, job_stats,
                           n_days_min=20, num_epochs_splits=3):
    epoch_edges_dct = {}
    eta_threshold_dct = defaultdict(list)
    eta_idxs_dct = defaultdict(list)
    num_objs_pass_eta = 0
    num_stars_pass_eta = 0
    for key, eta_arr in eta_dct.items():
        eta_arr = np.array(eta_arr)
        n_epochs = np.array(n_epochs_dct[key])
        stars_and_sources_idxs = np.array(idxs_dct[key])

        n_epochs_max = np.max(n_epochs)
        for i in range(num_epochs_splits, 0, -1):
            try:
                split_idx_arr, arr_bin_edges = evenly_split_sample(n_epochs,
                                                                   arr_min=n_days_min,
                                                                   arr_max=n_epochs_max+1,
                                                                   num_splits=i)
            except ValueError:
                pass
            else:
                break

        epoch_edges_dct[key] = arr_bin_edges

        for split_idx in split_idx_arr:
            eta_threshold = float(np.percentile(eta_arr[split_idx], 1))
            eta_threshold_dct[key].append(eta_threshold)

            eta_cond = np.where(eta_arr[split_idx] <= eta_threshold)[0]
            eta_idxs = stars_and_sources_idxs[split_idx][eta_cond]
            eta_idxs_dct[key].append(stars_and_sources_idxs[split_idx][eta_cond])

            num_objs_pass_eta += len(eta_idxs)
            num_stars_pass_eta += len(set([idx[0] for idx in eta_idxs]))

    job_stats['num_objs_pass_eta'] = num_objs_pass_eta
    job_stats['num_stars_pass_eta'] = num_stars_pass_eta
    job_stats['epoch_edges'] = epoch_edges_dct

    return eta_idxs_dct, eta_threshold_dct


def evenly_split_sample(arr, arr_min, arr_max, num_splits=3):
    # generate PDF of sample with a histogram
    bins = np.linspace(arr_min, arr_max, 10000)
    counts, _, bin_idxs = binned_statistic(arr, arr, 'count', bins=bins)

    # transform PDF into CDF
    cdf = np.cumsum(counts) / np.sum(counts)

    # find idxs for equal sections of the cdf
    cdf_edges = np.linspace(0, 1, num_splits + 1)
    cdf_idx_arr = []
    for i in range(num_splits):
        cdf_edge_low = cdf_edges[i]
        cdf_edge_high = cdf_edges[i+1]
        cdf_idx = set(np.where((cdf > cdf_edge_low) * (cdf <= cdf_edge_high))[0])
        if len(cdf_idx) == 0:
            raise ValueError('num_splits must be less than %i' % num_splits)
        cdf_idx_arr.append(cdf_idx)

    # instantiate empty arrays for idxs
    idx_arr = [[] for _ in range(num_splits)]

    # sort idxs into cdf buckets
    # for each element in the array...
    for i in range(len(arr)):
        # grab a bin idx...
        bin_idx = bin_idxs[i]
        # then compare that idxs to all of the CDF bins
        for j, cdf_idx in enumerate(cdf_idx_arr):
            # if the bin idx is within this CDF bin, add and break
            if bin_idx in cdf_idx:
                idx_arr[j].append(i)
                break

    # find the arr values at the edges
    arr_bin_edges = []
    for i in range(num_splits - 1):
        arr_bin_edges.append(int(np.max(arr[idx_arr[i]])))

    return idx_arr, arr_bin_edges


def construct_rf_idxs_dct(stars_and_sources, eta_idxs_dct, job_stats, obj_data):
    num_objs_pass_rf = 0
    num_stars_pass_rf = 0

    insert_db_id()
    ulens_con = catalog.ulens_con()
    rf_idxs_dct = defaultdict(list)
    for key, eta_idxs in eta_idxs_dct.items():
        for eta_idx in eta_idxs:
            rf_idxs = []
            for (i, j, k) in eta_idx:
                _, sources = stars_and_sources[i]
                source = sources[j]
                obj = source.zort_source.objects[k]
                rf_score = catalog.query_ps1_psc(obj.ra, obj.dec,
                                                 con=ulens_con)
                if rf_score is None or rf_score.rf_score >= RF_THRESHOLD:
                    rf_idxs.append((i, j, k))

                obj_key = (i, j, k)
                if rf_score is None:
                    obj_data[obj_key] = obj_data[obj_key]._replace(rf_score=0)
                else:
                    obj_data[obj_key] = obj_data[obj_key]._replace(rf_score=rf_score.rf_score)
            rf_idxs_dct[key].append(rf_idxs)

            num_objs_pass_rf += len(rf_idxs)
            num_stars_pass_rf += len(set([idx[0] for idx in rf_idxs]))
    ulens_con.close()
    remove_db_id()

    job_stats['num_objs_pass_rf'] = num_objs_pass_rf
    job_stats['num_stars_pass_rf'] = num_stars_pass_rf

    return rf_idxs_dct


def construct_eta_residual_idxs_dct(stars_and_sources, eta_threshold_dct, rf_idxs_dct, job_stats, obj_data):
    num_objs_pass_eta_residual = 0
    num_stars_pass_eta_residual = 0
    eta_residual_idxs_dct = defaultdict(list)
    for key in rf_idxs_dct.keys():
        rf_idxs_arr = rf_idxs_dct[key]
        eta_threshold_arr = eta_threshold_dct[key]
        for rf_idxs, eta_threshold in zip(rf_idxs_arr, eta_threshold_arr):
            eta_residual_idxs = []
            for (i, j, k) in rf_idxs:
                _, sources = stars_and_sources[i]
                source = sources[j]
                obj = source.zort_source.objects[k]
                hmjd = obj.lightcurve.hmjd
                mag = obj.lightcurve.mag
                magerr = obj.lightcurve.magerr
                eta_residual, fit_data = calculate_eta_on_daily_avg_residuals(hmjd, mag, magerr,
                                                                              return_fit_data=True)
                if eta_residual is not None and eta_residual > eta_threshold:
                    eta_residual_idxs.append((i, j, k))
                    obj_key = (i, j, k)
                    obj_data[obj_key] = obj_data[obj_key]._replace(eta_residual=eta_residual)
                    obj_data[obj_key] = obj_data[obj_key]._replace(fit_data=fit_data)

            eta_residual_idxs_dct[key].append(eta_residual_idxs)
            num_objs_pass_eta_residual += len(eta_residual_idxs)
            num_stars_pass_eta_residual += len(set([idx[0] for idx in eta_residual_idxs]))

    job_stats['num_objs_pass_eta_residual'] = num_objs_pass_eta_residual
    job_stats['num_stars_pass_eta_residual'] = num_stars_pass_eta_residual

    return eta_residual_idxs_dct


def extract_final_idxs(eta_residual_idxs_dct, eta_threshold_dct):
    final_idxs = []
    for key in eta_residual_idxs_dct.keys():
        eta_residual_idxs = eta_residual_idxs_dct[key]
        eta_thresholds = eta_threshold_dct[key]
        for eta_residual_idx, eta_threshold in zip(eta_residual_idxs, eta_thresholds):
            for i, j, k in eta_residual_idx:
                final_idxs.append((i, j, k, eta_threshold))
    return final_idxs


def assemble_candidates(stars_and_sources, eta_residual_idxs_dct, eta_threshold_dct, obj_data):
    candidates = []
    source_ids = []
    best_fit_stats = []

    final_idxs = extract_final_idxs(eta_residual_idxs_dct, eta_threshold_dct)
    unique_star_idxs = list(set([i for i, _, _, _ in final_idxs]))
    for star_idx in unique_star_idxs:
        star, sources = stars_and_sources[star_idx]
        source_obj_idxs = [(j, k) for i, j, k, _ in final_idxs if i == star_idx]
        eta_thresholds = [e for i, _, _, e in final_idxs if i == star_idx]
        num_objs_pass = len(source_obj_idxs)

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
        eta_threshold_best = eta_thresholds[idx_best]
        j, k = source_obj_idxs[idx_best]
        source = sources[j]
        source_ids.append(source.id)
        obj = source.zort_source.objects[k]

        obj_key = (star_idx, j, k)
        objectData = obj_data[obj_key]

        eta = objectData.eta
        rf_score = objectData.rf_score
        eta_residual = objectData.eta_residual
        fit_data = objectData.fit_data

        if fit_data:
            t_0, t_E, f_0, f_1, chi_squared_delta, chi_squared_flat, a_type = fit_data
        else:
            t_0 = None
            t_E = None
            f_0 = None
            f_1 = None
            chi_squared_delta = None
            chi_squared_flat = None
            a_type = None

        cand = Candidate(id=star.id,
                         source_id_arr=source_id_arr,
                         filter_id_arr=filter_id_arr,
                         ra=star.ra,
                         dec=star.dec,
                         ingest_job_id=star.ingest_job_id,
                         eta_best=eta,
                         rf_score_best=rf_score,
                         eta_residual_best=eta_residual,
                         eta_threshold_best=eta_threshold_best,
                         t_E_best=t_E,
                         t_0_best=t_0,
                         f_0_best=f_0,
                         f_1_best=f_1,
                         a_type_best=a_type,
                         chi_squared_flat_best=chi_squared_flat,
                         chi_squared_delta_best=chi_squared_delta,
                         idx_best=idx_best,
                         num_objs_pass=num_objs_pass)
        candidates.append(cand)

        fit_filter = obj.color
        best_fit_stats.append((fit_filter, t_E, t_0, f_0, f_1, a_type, chi_squared_flat, chi_squared_delta))

    return candidates, source_ids, best_fit_stats


def filter_stars_to_candidates(source_job_id, stars_and_sources,
                               n_days_min=20, num_epochs_splits=3):
    job_stats = {}
    obj_data = {}
    logger.info(f'Job {source_job_id}: Calculating eta')
    eta_dct, idxs_dct, n_epochs_dct = construct_eta_dct(stars_and_sources, job_stats, obj_data,
                                                        n_days_min=n_days_min)
    logger.info(f'Job {source_job_id}: '
                f'{job_stats["num_stars"]} Stars | '
                f'{job_stats["num_sources"]} Sources | '
                f'{job_stats["num_objs"]} Objects | '
                f'{job_stats["num_objs_pass_n_days"]} Objects Past Days Cuts')

    eta_idxs_dct, eta_threshold_dct = construct_eta_idxs_dct(eta_dct, idxs_dct,
                                                             n_epochs_dct, job_stats,
                                                             n_days_min=n_days_min,
                                                             num_epochs_splits=num_epochs_splits)
    logger.info(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_eta"]} stars pass eta cut | '
                f'{job_stats["num_objs_pass_eta"]} objects pass eta cut')

    rf_idxs_dct = construct_rf_idxs_dct(stars_and_sources, eta_idxs_dct, job_stats, obj_data)
    logger.info(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_rf"]} stars pass rf_score cut | '
                f'{job_stats["num_objs_pass_rf"]} objects pass rf_score cut')

    eta_residual_idxs_dct = construct_eta_residual_idxs_dct(stars_and_sources,
                                                            eta_threshold_dct,
                                                            rf_idxs_dct, job_stats, obj_data)
    logger.info(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_eta_residual"]} source pass eta_residual cut | '
                f'{job_stats["num_objs_pass_eta_residual"]} objects pass eta_residual cut')

    logger.info(f'Job {source_job_id}: Assembling candidates')
    candidates, source_ids, best_fit_stats = assemble_candidates(stars_and_sources, eta_residual_idxs_dct,
                                                                 eta_threshold_dct, obj_data)
    job_stats['num_candidates'] = len(candidates)
    job_stats['eta_thresholds'] = eta_threshold_dct
    return candidates, job_stats, source_ids, best_fit_stats


def upload_candidates(candidates, source_ids, best_fit_stats):
    insert_db_id()  # get permission to make a db connection
    for cand in candidates:
        db.session.add(cand)

    # grab sources from db and hash them in dictionary
    source_dbs = db.session.query(Source).filter(Source.id.in_(source_ids)).all()
    source_db_dct = {}
    for source_db in source_dbs:
        source_db_dct[source_db.id] = source_db

    for source_id, best_fit_stat in zip(source_ids, best_fit_stats):
        source_db = source_db_dct[source_id]
        fit_filter, t_E, t_0, f_0, f_1, a_type, chi_squared_flat, chi_squared_delta = best_fit_stat
        source_db.fit_filter = fit_filter
        source_db.fit_t_0 = t_0
        source_db.fit_t_E = t_E
        source_db.fit_f_0 = f_0
        source_db.fit_f_1 = f_1
        source_db.fit_a_type = a_type
        source_db.fit_chi_squared_flat = chi_squared_flat
        source_db.fit_chi_squared_delta = chi_squared_delta
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
    job.num_objs_pass_n_days = job_stats['num_objs_pass_n_days']
    job.num_objs_pass_eta = job_stats['num_objs_pass_eta']
    job.num_stars_pass_eta = job_stats['num_stars_pass_eta']
    job.num_objs_pass_rf = job_stats['num_objs_pass_rf']
    job.num_stars_pass_rf = job_stats['num_stars_pass_rf']
    job.num_objs_pass_eta_residual = job_stats['num_objs_pass_eta_residual']
    job.num_stars_pass_eta_residual = job_stats['num_stars_pass_eta_residual']
    job.epoch_edges = job_stats['epoch_edges']
    job.eta_thresholds = job_stats['eta_thresholds']
    job.num_candidates = job_stats['num_candidates']
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
            candidates, job_stats, source_ids, best_fit_stats = filter_stars_to_candidates(
                source_job_id, stars_and_sources)
            num_candidates = len(candidates)
            logger.info(f'Job {source_job_id}: Uploading {num_candidates} candidates')
            if num_candidates > 0:
                upload_candidates(candidates, source_ids, best_fit_stats)
            logger.info(f'Job {source_job_id}: Processing complete')
        else:
            logger.info(f'Job {source_job_id}: No stars, skipping process')
            job_stats = {'num_stars': 0,
                         'num_sources': 0,
                         'num_objs': 0,
                         'num_objs_pass_n_days': 0,
                         'num_objs_pass_eta': 0,
                         'num_stars_pass_eta': 0,
                         'num_objs_pass_rf': 0,
                         'num_stars_pass_rf': 0,
                         'num_objs_pass_eta_residual': 0,
                         'num_stars_pass_eta_residual': 0,
                         'epoch_edges': {},
                         'eta_thresholds': {},
                         'num_candidates': 0}

        finish_job(source_job_id, job_stats)
        logger.info(f'Job {source_job_id}: Job complete')

        if single_job:
            return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_stars()
