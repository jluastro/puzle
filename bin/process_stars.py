#! /usr/bin/env python
"""
process_stars.py
"""
import time
import os
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import binned_statistic
from collections import defaultdict, namedtuple
import pickle

from puzle.models import Source, StarIngestJob, Star, StarProcessJob, Candidate
from puzle.utils import fetch_job_enddate, return_DR4_dir, get_logger
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.stats import calculate_eta_on_daily_avg, \
    RF_THRESHOLD, calculate_eta_on_daily_avg_residuals
from puzle import catalog
from puzle import db

logger = get_logger(__name__)

ObjectData = namedtuple('ObjectData', 'eta eta_residual rf_score fit_data eta_threshold_low eta_threshold_high')


def fetch_job():
    insert_db_id()  # get permission to make a db connection

    slurm_job_id = os.getenv('SLURM_JOB_ID', 0)
    print(f'ID {slurm_job_id}: Fetching job')

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
        rank = comm.Get_rank()
    else:
        rank = 0

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


def fetch_stars_and_sources(source_job_id):
    DR4_dir = return_DR4_dir()
    dir = '%s/stars_%s' % (DR4_dir, str(source_job_id)[:3])

    if not os.path.exists(dir):
        logger.error('Source directory missing for %i' % source_job_id)
        import glob
        fis = glob.glob('%s/*' % DR4_dir)
        for fi in fis:
            logger.error(fi)
        return

    fname = f'{dir}/stars.{source_job_id:06}.txt'
    if not os.path.exists(fname):
        logger.error('Source file missing for %i' % source_job_id)
        return

    sources_fname = fname.replace('star', 'source')
    sources_map_fname = sources_fname.replace('.txt', '.sources_map')
    sources_map = pickle.load(open(sources_map_fname, 'rb'))

    lines_star = open(fname, 'r').readlines()[1:]
    f_sources = open(sources_fname, 'r')

    stars_and_sources = {}
    for i, line_star in enumerate(lines_star):
        star = csv_line_to_star_and_sources(line_star)
        source_arr = []
        for source_id in star.source_ids:
            f_sources.seek(sources_map[source_id])
            line_source = f_sources.readline()
            source_arr.append(csv_line_to_source(line_source))
        stars_and_sources[star.id] = (star, source_arr)

    return stars_and_sources


def construct_eta_dct(stars_and_sources, job_stats, obj_data, n_days_min=20):
    num_stars = 0
    num_objs = 0
    num_objs_pass_n_days = 0
    idxs_dct = defaultdict(list)
    eta_dct = defaultdict(list)
    n_days_dct = defaultdict(list)
    star_ids_pass_n_days = set()
    for star_id, (star, sources) in stars_and_sources.items():
        num_stars += 1
        for j, source in enumerate(sources):
            for k, obj in enumerate(source.zort_source.objects):
                num_objs += 1
                n_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
                if n_days < n_days_min:
                    continue
                num_objs_pass_n_days += 1
                star_ids_pass_n_days.add(star_id)
                eta = calculate_eta_on_daily_avg(obj.lightcurve.hmjd,
                                                 obj.lightcurve.mag)

                key = '%i_%i' % (obj.fieldid, obj.filterid)
                idxs_dct[key].append((star_id, j, k))
                eta_dct[key].append(eta)
                n_days_dct[key].append(n_days)

                obj_key = (star_id, j, k)
                objectData = ObjectData(eta=eta, eta_residual=None,
                                        rf_score=None, fit_data=None,
                                        eta_threshold_low=None,
                                        eta_threshold_high=None)
                obj_data[obj_key] = objectData
    num_stars_pass_n_days = len(star_ids_pass_n_days)

    job_stats['num_stars'] = num_stars
    job_stats['num_stars_pass_n_days'] = num_stars_pass_n_days
    job_stats['num_objs'] = num_objs
    job_stats['num_objs_pass_n_days'] = num_objs_pass_n_days

    return eta_dct, idxs_dct, n_days_dct


def construct_eta_idxs_dct(eta_dct, idxs_dct, n_days_dct, job_stats, obj_data,
                           n_days_min=20, num_epochs_splits=3):
    epoch_edges_dct = {}
    eta_threshold_low_dct = defaultdict(list)
    eta_threshold_high_dct = defaultdict(list)
    eta_idxs_dct = defaultdict(list)
    num_objs_pass_eta = 0
    num_stars_pass_eta = 0
    for key, eta_arr in eta_dct.items():
        eta_arr = np.array(eta_arr)
        n_days = np.array(n_days_dct[key])
        stars_and_sources_idxs = idxs_dct[key]

        n_days_max = np.max(n_days)
        for i in range(num_epochs_splits, 0, -1):
            try:
                split_idx_arr, arr_bin_edges = evenly_split_sample(n_days,
                                                                   arr_min=n_days_min,
                                                                   arr_max=n_days_max+1,
                                                                   num_splits=i)
            except ValueError:
                pass
            else:
                break

        epoch_edges_dct[key] = arr_bin_edges

        for split_idx in split_idx_arr:
            eta_threshold_low = float(np.percentile(eta_arr[split_idx], 1))
            eta_threshold_low_dct[key].append(eta_threshold_low)

            eta_threshold_high = float(np.percentile(eta_arr[split_idx], 90))
            eta_threshold_high_dct[key].append(eta_threshold_high)

            eta_cond = np.where(eta_arr[split_idx] <= eta_threshold_low)[0]
            eta_idxs_split = [stars_and_sources_idxs[idx] for idx in split_idx]
            eta_idxs = [eta_idxs_split[idx] for idx in eta_cond]
            eta_idxs_dct[key].append(eta_idxs)

            for i, j, k in eta_idxs:
                obj_key = (i, j, k)
                obj_data[obj_key] = obj_data[obj_key]._replace(eta_threshold_low=eta_threshold_low,
                                                               eta_threshold_high=eta_threshold_high)

            num_objs_pass_eta += len(eta_idxs)
            num_stars_pass_eta += len(set([idx[0] for idx in eta_idxs]))

    job_stats['num_objs_pass_eta'] = num_objs_pass_eta
    job_stats['num_stars_pass_eta'] = num_stars_pass_eta
    job_stats['epoch_edges'] = epoch_edges_dct

    return eta_idxs_dct, eta_threshold_low_dct, eta_threshold_high_dct


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

    ps1_psc_dct = {}
    rf_idxs_dct = defaultdict(list)
    for key, eta_idxs in eta_idxs_dct.items():
        for eta_idx in eta_idxs:
            rf_idxs = []
            for (i, j, k) in eta_idx:
                _, sources = stars_and_sources[i]
                source = sources[j]
                obj = source.zort_source.objects[k]
                rf_score = catalog.query_ps1_psc_on_disk(obj.ra, obj.dec,
                                                         ps1_psc_dct=ps1_psc_dct)
                if rf_score is None:
                    rf_idxs.append((i, j, k))
                elif rf_score.rf_score >= RF_THRESHOLD:
                    obj_key = (i, j, k)
                    obj_data[obj_key] = obj_data[obj_key]._replace(rf_score=rf_score.rf_score)
                    rf_idxs.append((i, j, k))

            rf_idxs_dct[key].append(rf_idxs)

            num_objs_pass_rf += len(rf_idxs)
            num_stars_pass_rf += len(set([idx[0] for idx in rf_idxs]))

    job_stats['num_objs_pass_rf'] = num_objs_pass_rf
    job_stats['num_stars_pass_rf'] = num_stars_pass_rf

    return rf_idxs_dct


def construct_eta_residual_idxs_dct(stars_and_sources, eta_threshold_high_dct, rf_idxs_dct, job_stats, obj_data):
    num_objs_pass_eta_residual = 0
    num_stars_pass_eta_residual = 0
    eta_residual_idxs_dct = defaultdict(list)
    for key in rf_idxs_dct.keys():
        rf_idxs_arr = rf_idxs_dct[key]
        eta_threshold_arr = eta_threshold_high_dct[key]
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


def extract_final_idxs(eta_residual_idxs_dct):
    final_idxs = []
    for eta_residual_idxs in eta_residual_idxs_dct.values():
        for eta_residual_idx in eta_residual_idxs:
            for i, j, k in eta_residual_idx:
                final_idxs.append((i, j, k))
    return final_idxs


def assemble_candidates(stars_and_sources, eta_residual_idxs_dct, obj_data):
    candidates = []
    source_id_to_cand_id_dct = {}
    source_ids = []
    fit_stats_best = []

    final_idxs = extract_final_idxs(eta_residual_idxs_dct)
    unique_star_idxs = list(set([i for i, _, _ in final_idxs]))
    for star_idx in unique_star_idxs:
        # extract the passing indices
        source_obj_idxs_pass = [(j, k) for i, j, k in final_idxs if i == star_idx]
        num_objs_pass = len(source_obj_idxs_pass)

        # extract all indices, and note which ones are passing
        star, sources = stars_and_sources[star_idx]
        source_id_arr = []
        color_arr = []
        pass_arr = []
        n_days_arr = []
        source_obj_idxs_tot = []
        for j, source in enumerate(sources):
            for k, obj in enumerate(source.zort_source.objects):
                source_id_arr.append(source.id)
                source_id_to_cand_id_dct[source.id] = star.id
                color_arr.append(obj.color)
                source_obj_idxs_tot.append((j, k))
                if (j, k) in source_obj_idxs_pass:
                    n_days = len(np.unique(np.floor(obj.lightcurve.hmjd)))
                    n_days_arr.append(n_days)
                    pass_arr.append(True)
                else:
                    pass_arr.append(False)
                    n_days_arr.append(0)
        num_objs_tot = len(source_obj_idxs_tot)

        idx_best = int(np.argmax(n_days_arr))
        j, k = source_obj_idxs_tot[idx_best]
        source = sources[j]
        source_ids.append(source.id)
        obj = source.zort_source.objects[k]

        obj_key = (star_idx, j, k)
        objectData = obj_data[obj_key]

        eta = objectData.eta
        rf_score = objectData.rf_score
        eta_residual = objectData.eta_residual
        fit_data = objectData.fit_data
        eta_threshold_low = objectData.eta_threshold_low
        eta_threshold_high = objectData.eta_threshold_high

        # if fit_data is None, then this should not be a valid "best" object!
        # assert fit_data is not None
        if fit_data is not None:
            t_0, t_E, f_0, f_1, chi_squared_delta, chi_squared_flat, a_type = fit_data
        else:
            t_0, t_E, f_0, f_1, chi_squared_delta, chi_squared_flat, a_type = None, None, None, None, None, None, None

        cand = Candidate(id=star.id,
                         source_id_arr=source_id_arr,
                         color_arr=color_arr,
                         pass_arr=pass_arr,
                         ra=star.ra,
                         dec=star.dec,
                         ingest_job_id=star.ingest_job_id,
                         eta_best=eta,
                         rf_score_best=rf_score,
                         eta_residual_best=eta_residual,
                         eta_threshold_low_best=eta_threshold_low,
                         eta_threshold_high_best=eta_threshold_high,
                         t_E_best=t_E,
                         t_0_best=t_0,
                         f_0_best=f_0,
                         f_1_best=f_1,
                         a_type_best=a_type,
                         chi_squared_flat_best=chi_squared_flat,
                         chi_squared_delta_best=chi_squared_delta,
                         idx_best=idx_best,
                         num_objs_pass=num_objs_pass,
                         num_objs_tot=num_objs_tot)
        candidates.append(cand)

        fit_filter = obj.color
        fit_stats_best.append((fit_filter, t_E, t_0, f_0, f_1, a_type, chi_squared_flat, chi_squared_delta))

    return candidates, source_ids, fit_stats_best, source_id_to_cand_id_dct


def filter_stars_to_candidates(source_job_id, stars_and_sources,
                               n_days_min=20, num_epochs_splits=3):
    job_stats = {}
    obj_data = {}
    print(f'Job {source_job_id}: Calculating eta')
    eta_dct, idxs_dct, n_days_dct = construct_eta_dct(stars_and_sources, job_stats, obj_data,
                                                        n_days_min=n_days_min)
    print(f'Job {source_job_id}: '
                f'{job_stats["num_stars"]} Stars | '
                f'{job_stats["num_stars_pass_n_days"]} Stars Past Days Cuts | '
                f'{job_stats["num_objs"]} Objects | '
                f'{job_stats["num_objs_pass_n_days"]} Objects Past Days Cuts')

    eta_idxs_dct, eta_threshold_low_dct, eta_threshold_high_dct = construct_eta_idxs_dct(
        eta_dct, idxs_dct, n_days_dct, job_stats, obj_data,
        n_days_min=n_days_min, num_epochs_splits=num_epochs_splits)
    print(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_eta"]} stars pass eta cut | '
                f'{job_stats["num_objs_pass_eta"]} objects pass eta cut')

    rf_idxs_dct = construct_rf_idxs_dct(stars_and_sources, eta_idxs_dct, job_stats, obj_data)
    print(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_rf"]} stars pass rf_score cut | '
                f'{job_stats["num_objs_pass_rf"]} objects pass rf_score cut')

    eta_residual_idxs_dct = construct_eta_residual_idxs_dct(stars_and_sources,
                                                            eta_threshold_high_dct,
                                                            rf_idxs_dct, job_stats, obj_data)
    print(f'Job {source_job_id}: '
                f'{job_stats["num_stars_pass_eta_residual"]} source pass eta_residual cut | '
                f'{job_stats["num_objs_pass_eta_residual"]} objects pass eta_residual cut')

    print(f'Job {source_job_id}: Assembling candidates')
    candidates, source_ids, fit_stats_best, source_id_to_cand_id_dct = assemble_candidates(
        stars_and_sources, eta_residual_idxs_dct, obj_data)
    job_stats['num_candidates'] = len(candidates)
    job_stats['eta_thresholds_low'] = eta_threshold_low_dct
    job_stats['eta_thresholds_high'] = eta_threshold_high_dct
    return candidates, job_stats, source_ids, fit_stats_best, source_id_to_cand_id_dct


def upload_candidates(candidates, source_ids, fit_stats_best, source_id_to_cand_id_dct):
    insert_db_id()  # get permission to make a db connection
    for cand in candidates:
        db.session.add(cand)

    # grab sources from db and hash them in dictionary
    source_ids_tot = list(source_id_to_cand_id_dct.keys())
    source_dbs = db.session.query(Source).filter(Source.id.in_(source_ids_tot)).all()
    source_db_dct = {}
    for source_db in source_dbs:
        source_db_dct[source_db.id] = source_db
        source_db.cand_id = source_id_to_cand_id_dct[source_db.id]

    for source_id, fit_stat in zip(source_ids, fit_stats_best):
        source_db = source_db_dct[source_id]
        fit_filter, t_E, t_0, f_0, f_1, a_type, chi_squared_flat, chi_squared_delta = fit_stat
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
    job.num_stars_pass_n_days = job_stats['num_stars_pass_n_days']
    job.num_objs = job_stats['num_objs']
    job.num_objs_pass_n_days = job_stats['num_objs_pass_n_days']
    job.num_objs_pass_eta = job_stats['num_objs_pass_eta']
    job.num_stars_pass_eta = job_stats['num_stars_pass_eta']
    job.num_objs_pass_rf = job_stats['num_objs_pass_rf']
    job.num_stars_pass_rf = job_stats['num_stars_pass_rf']
    job.num_objs_pass_eta_residual = job_stats['num_objs_pass_eta_residual']
    job.num_stars_pass_eta_residual = job_stats['num_stars_pass_eta_residual']
    job.epoch_edges = job_stats['epoch_edges']
    job.eta_thresholds_low = job_stats['eta_thresholds_low']
    job.eta_thresholds_high = job_stats['eta_thresholds_high']
    job.num_candidates = job_stats['num_candidates']
    db.session.commit()
    db.session.close()
    remove_db_id()  # release permission for this db connection


def process_stars(source_job_id):
    stars_and_sources = fetch_stars_and_sources(source_job_id)

    num_stars = len(stars_and_sources)
    if num_stars > 0:
        print(f'Job {source_job_id}: Processing {num_stars} stars')
        candidates, job_stats, source_ids, fit_stats_best, source_id_to_cand_id_dct = filter_stars_to_candidates(
            source_job_id, stars_and_sources)
        num_candidates = len(candidates)
        print(f'Job {source_job_id}: Uploading {num_candidates} candidates')
        if num_candidates > 0:
            upload_candidates(candidates, source_ids, fit_stats_best, source_id_to_cand_id_dct)
        print(f'Job {source_job_id}: Processing complete')
    else:
        print(f'Job {source_job_id}: No stars, skipping process')
        job_stats = {'num_stars': 0,
                     'num_stars_pass_n_days': 0,
                     'num_objs': 0,
                     'num_objs_pass_n_days': 0,
                     'num_objs_pass_eta': 0,
                     'num_stars_pass_eta': 0,
                     'num_objs_pass_rf': 0,
                     'num_stars_pass_rf': 0,
                     'num_objs_pass_eta_residual': 0,
                     'num_stars_pass_eta_residual': 0,
                     'epoch_edges': {},
                     'eta_thresholds_low': {},
                     'eta_thresholds_high': {},
                     'num_candidates': 0}

    finish_job(source_job_id, job_stats)
    print(f'Job {source_job_id}: Job complete')


def process_stars_script(shutdown_time=10, single_job=False):
    try:
        job_enddate = fetch_job_enddate()
    except FileNotFoundError:
        job_enddate = None
    if job_enddate:
        script_enddate = job_enddate - timedelta(minutes=shutdown_time)
        print('Script End Date: %s' % script_enddate)

    while True:

        source_job_id = fetch_job()
        if source_job_id is None:
            return

        if job_enddate and datetime.now() >= script_enddate:
            print(f'Within {shutdown_time} minutes of job end, '
                        f'shutting down...')
            reset_job(source_job_id)
            time.sleep(2 * 60 * shutdown_time)
            return

        process_stars(source_job_id)

        if single_job:
            return


if __name__ == '__main__':
    process_stars_script()
