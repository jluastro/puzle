#! /usr/bin/env python
"""
study_mroz_sources.py
"""

import pickle
import os
import numpy as np

from puzle.utils import return_DR5_dir
from puzle.models import SourceIngestJob, StarProcessJob, \
    Star, Source, Candidate
from puzle.stats import calculate_eta_on_daily_avg, \
    RF_THRESHOLD, calculate_eta_on_daily_avg_residuals
from puzle import catalog
from puzle.utils import return_figures_dir
from puzle import db


def return_source_job_id_and_eta_thresholds(ra, dec):
    job = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id,
               SourceIngestJob.ra_start < ra,
               SourceIngestJob.ra_end > ra,
               SourceIngestJob.dec_start < dec,
               SourceIngestJob.dec_end > dec).\
        with_entities(SourceIngestJob.id,
                      StarProcessJob.epoch_edges,
                      StarProcessJob.eta_thresholds_low).\
        all()
    assert len(job) == 1
    job = job[0]

    source_job_id = job[0]
    epoch_edges = job[1]
    eta_thresholds = job[2]

    return source_job_id, epoch_edges, eta_thresholds


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


def csv_line_to_source(line, lightcurve_file_pointers):
    attrs = line.replace('\n', '').split(',')

    filename = attrs[7]
    if filename in lightcurve_file_pointers:
        lightcurve_file_pointer = lightcurve_file_pointers[filename]
    else:
        lightcurve_file_pointer = open(filename, 'r')
        lightcurve_file_pointers[filename] = lightcurve_file_pointer

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
                    ingest_job_id=int(attrs[10]),
                    lightcurve_file_pointer=lightcurve_file_pointer)
    return source


def fetch_star(source_job_id, ra, dec):
    DR5_dir = return_DR5_dir()
    dir = '%s/stars_%s' % (DR5_dir, str(source_job_id)[:3])
    fname = f'{dir}/stars.{source_job_id:06}.txt'

    sources_fname = fname.replace('star', 'source')
    sources_map_fname = sources_fname.replace('.txt', '.sources_map')
    sources_map = pickle.load(open(sources_map_fname, 'rb'))

    lines_star = open(fname, 'r').readlines()[1:]
    f_sources = open(sources_fname, 'r')

    stars_and_sources = {}
    lightcurve_file_pointers = {}
    for i, line_star in enumerate(lines_star):
        star = csv_line_to_star_and_sources(line_star)
        if np.hypot(star.ra - ra, star.dec - dec) > 2 / 3600.:
            continue
        source_arr = []
        for source_id in star.source_ids:
            f_sources.seek(sources_map[source_id])
            line_source = f_sources.readline()
            source = csv_line_to_source(line_source, lightcurve_file_pointers)
            # initialize lightcurves with file pointers
            for obj in source.zort_source.objects:
                _ = obj.lightcurve.nepochs
            source_arr.append(source)
        stars_and_sources[star.id] = (star, source_arr)

    f_sources.close()

    for file in lightcurve_file_pointers.values():
        file.close()

    return stars_and_sources


def hmjd_to_n_days(hmjd):
    hmjd_round = np.round(hmjd)
    hmjd_unique = set(hmjd_round)
    n_days = len(hmjd_unique)
    return n_days


def return_best_obj(cand):
    idx = cand.idx_best
    source_id = cand.source_id_arr[idx]
    source = Source.query.filter(Source.id==source_id).first()
    color = cand.color_arr[idx]
    obj = getattr(source.zort_source, f'object_{color}')
    return obj


def return_outcome(ztf_id, ra, dec):
    figures_dir = return_figures_dir()
    mroz_figures_dir = f'{figures_dir}/mroz'
    if not os.path.exists(mroz_figures_dir):
        os.makedirs(mroz_figures_dir)
    print('Searching %s (ra, dec) = (%.3f, %.3f)' % (ztf_id, ra, dec))

    cone_filter = Candidate.cone_search(ra, dec)
    cands = db.session.query(Candidate).filter(cone_filter).all()
    if len(cands) == 1:
        print('-- in database')
        slope, eta_thresh = 3.57, 0.6
        cand = cands[0]
        obj = return_best_obj(cand)
        color = obj.color
        cand_id = cand.id.replace('_', '-')
        if cand.eta_best <= eta_thresh and cand.eta_residual_best > cand.eta_best * slope:
            print('-- passes cuts')
            obj.plot_lightcurve(filename=f'{mroz_figures_dir}/{ztf_id}_{cand_id}_{color}_lc_in_db_passes_cuts.png')
            return 'in database - passes cuts'
        else:
            print('-- no passes cuts')
            obj.plot_lightcurve(filename=f'{mroz_figures_dir}/{ztf_id}_{cand_id}_{color}_lc_in_db_no_passes_cuts.png')
            return 'in database - no passes cuts'
    elif len(cands) > 1:
        print('-- multiple in database')
        return 'multiple in database'

    n_days_min = 50
    source_job_id, epoch_edges, eta_thresholds = return_source_job_id_and_eta_thresholds(ra, dec)
    stars_and_sources = fetch_star(source_job_id, ra, dec)
    star_id, (star, sources) = list(stars_and_sources.items())[0]

    n_objs = 0
    n_days_idxs = []
    for j, source in enumerate(sources):
        for k, obj in enumerate(source.zort_source.objects):
            n_objs += 1
            color = obj.color
            source_id = source.id.replace('_', '-')
            hmjd = obj.lightcurve.hmjd
            n_days = hmjd_to_n_days(hmjd)
            if n_days >= n_days_min:
                n_days_idxs.append((j, k))
            else:
                obj.plot_lightcurve(filename=f'{ztf_id}_{source_id}_{color}_lc_days.png')

    print('-- num_objs = %i' % n_objs)
    print('-- num_objs_pass_days = %i' % len(n_days_idxs))

    if len(n_days_idxs) == 0:
        return 'days'

    eta_threshold_idxs = []
    for j, k in n_days_idxs:
        source = sources[j]
        obj = source.zort_source.objects[k]
        color = obj.color
        source_id = source.id.replace('_', '-')
        n_days = hmjd_to_n_days(obj.lightcurve.hmjd)
        key = '%i_%i' % (obj.fieldid, obj.filterid)

        threshold_idx = np.searchsorted(epoch_edges[key], n_days)
        eta_threshold = eta_thresholds[key][threshold_idx]
        eta = calculate_eta_on_daily_avg(obj.lightcurve.hmjd,
                                         obj.lightcurve.mag)
        if eta <= eta_threshold:
            eta_threshold_idxs.append((j, k))
        else:
            obj.plot_lightcurve(filename=f'{mroz_figures_dir}/{ztf_id}_{source_id}_{color}_lc_eta.png')

    print('-- num_objs_pass_eta = %i' % len(eta_threshold_idxs))
    if len(eta_threshold_idxs) == 0:
        return 'eta'

    ps1_psc_dct = {}
    rf_idxs = []
    for j, k in eta_threshold_idxs:
        source = sources[j]
        obj = source.zort_source.objects[k]
        color = obj.color
        source_id = source.id.replace('_', '-')
        rf_score = catalog.query_ps1_psc_on_disk(obj.ra, obj.dec,
                                                 ps1_psc_dct=ps1_psc_dct)
        if rf_score is None or rf_score.rf_score >= RF_THRESHOLD:
            rf_idxs.append((j, k))
        else:
            obj.plot_lightcurve(filename=f'{mroz_figures_dir}/{ztf_id}_{source_id}_{color}_lc_rf.png')

    print('-- num_objs_pass_rf = %i' % len(rf_idxs))
    if len(rf_idxs) == 0:
        return 'rf'

    eta_residual_idxs = []
    for j, k in eta_threshold_idxs:
        source = sources[j]
        obj = source.zort_source.objects[k]
        color = obj.color
        source_id = source.id.replace('_', '-')
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr
        eta_residual, fit_data = calculate_eta_on_daily_avg_residuals(hmjd, mag, magerr,
                                                                      return_fit_data=True)
        if eta_residual is not None and not np.isnan(eta_residual):
            eta_residual_idxs.append((j, k))
        else:
            obj.plot_lightcurve(filename=f'{mroz_figures_dir}/{ztf_id}_{source_id}_{color}_lc_eta_residual.png')

    print('-- num_objs_pass_eta_residual = %i' % len(eta_residual_idxs))
    if len(eta_residual_idxs) == 0:
        return 'eta_residual'

    print('-- missing from database')
    return 'missing from db'


def calculate_outcomes():
    radec_mroz = np.array([
        ['ZTF18aatnfdf', 286.633211, 32.248996]
        , ['ZTF18aazdbym', 290.784286, 7.810517]
        , ['ZTF18aaztjyd', 326.173116, 59.377872]
        , ['ZTF18aazwhtw', 339.955528, 51.647223]
        , ['ZTF18abaqxrt', 290.617225, 1.706486]
        , ['ZTF18abhxjmj', 284.029167, 13.152260]
        , ['ZTF18ablrbk', 271.850400, -10.314477]
        , ['ZTF18ablrdcc', 271.439120, -12.014556]
        , ['ZTF18ablruzq', 284.338291, 11.433438]
        , ['ZTF18abmoxlq', 285.984027, -13.929477]
        , ['ZTF18abnbmsr', 307.149376, 22.83047]
        , ['ZTF18abqawpf', 287.113964, 1.531903]
        , ['ZTF18abqazwfa', 285.134471, 30.511120]
        , ['ZTF18abqbeqv', 279.578723, 7.837854]
        , ['ZTF18absrqlr', 307.149376, 22.830478]
        , ['ZTF18abtnvsg', 291.019150, 20.478976]
        , ['ZTF18acskgwu', 76.632447, 8.425664]
        , ['ZTF19aabbuqn', 48.694244, 62.343390]
        , ['ZTF19aaekacq', 279.404621, 11.200516]
        , ['ZTF19aainwvb', 55.197569, 57.955805]
        , ['ZTF19aamlgyh', 289.114418, 26.653532]
        , ['ZTF19aamrjmu', 280.734529, 32.873054]
        , ['ZTF19aaonska', 273.900566, -2.256985]
        , ['ZTF19aaprbng', 274.913476, 0.590991]
        , ['ZTF19aatudnja', 290.663294, 19.550373]
        , ['ZTF19aatwaux', 258.208411, -27.182057]
        , ['ZTF19aavisrq', 297.706148, 34.637344]
        , ['ZTF19aavndrc', 281.836951, -4.338099]
        , ['ZTF19aavnrqt', 309.034132, 32.720880]
        , ['ZTF19aaxsdqz', 283.497170, -1.152267]])

    outcome_arr = []
    for ztf_id, ra, dec in radec_mroz:
        ra, dec = float(ra), float(dec)
        outcome = return_outcome(ztf_id, ra, dec)
        outcome_arr.append((ztf_id, outcome))

    in_db_passes = [ztf_id for ztf_id, outcome in outcome_arr if 'in database - passes cuts'==outcome]
    in_db_no_passes = [ztf_id for ztf_id, outcome in outcome_arr if 'in database - no passes cuts'==outcome]
    cut_due_to_days = [ztf_id for ztf_id, outcome in outcome_arr if 'days'==outcome]
    cut_due_to_eta = [ztf_id for ztf_id, outcome in outcome_arr if 'eta'==outcome]
    cut_due_to_rf = [ztf_id for ztf_id, outcome in outcome_arr if 'rf'==outcome]
    cut_due_to_eta_residual = [ztf_id for ztf_id, outcome in outcome_arr if 'eta_residual'==outcome]

    print('%i total mroz sources' % len(radec_mroz))
    print('-- %i sources in database' % (len(in_db_passes)+len(in_db_no_passes)))
    print('---- %i sources in database that WILL pass Level 2 cuts' % len(in_db_passes))
    print('---- %i sources in database that WILL NOT pass Level 2 cuts' % len(in_db_no_passes))
    print('-- %i sources cut on Level 1 due to days' % len(cut_due_to_days))
    print('-- %i sources cut on Level 1 due to eta' % len(cut_due_to_eta))
    print('-- %i sources cut on Level 1 due to rf' % len(cut_due_to_rf))
    print('-- %i sources cut on Level 1 due to eta residual' % len(cut_due_to_eta_residual))


if __name__ == '__main__':
    calculate_outcomes()
