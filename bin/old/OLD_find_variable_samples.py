#! /usr/bin/env python
"""
OLD_find_variable_samples.py
"""
import random
import os
import glob
import numpy as np
from zort.lightcurveFile import LightcurveFile, locate_sources_by_radec
from zort.source import create_source_from_object
from puzle.models import Source, SourceIngestJob
from puzle import db


def _return_obj_lightcurve_data(obj):
    if obj is None:
        hmjd = []
        mag = []
        magerr = []
    else:
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr
    return hmjd, mag, magerr


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def find_variable_samples(N_candidates=20000, N_samples=20):
    folder = 'variable_sample/variable'
    if not os.path.exists(folder):
        os.makedirs(folder)

    field_id_arr = [282, 283, 333, 334, 539, 590]
    for field_id in field_id_arr:
        filename = glob.glob('field%06d_*txt' % field_id)[0]

        object_filename = filename.replace('.txt', '.objects')

        n_objects = file_len(object_filename)
        process_size = (n_objects // N_candidates) + 1

        lightcurveFile = LightcurveFile(filename, proc_rank=0, proc_size=process_size)
        eta_arr = []
        obj_arr = []
        for obj in lightcurveFile:
            if obj.lightcurve.nepochs < 100:
                continue
            hmjd = obj.lightcurve.hmjd
            hmjd_min = int(np.min(hmjd))
            hmjd_max = int(np.max(hmjd))
            if hmjd_max - hmjd_min < 10:
                continue

            flux = []
            t_arr = []
            for t in np.arange(hmjd_min, hmjd_max + 1):
                cond = np.floor(hmjd) == t
                if np.sum(cond) == 0:
                    continue
                flux.append(np.median(obj.lightcurve.flux[cond]))
                t_arr.append(t)
            flux = np.array(flux)

            eta = np.sum((flux[1:] - flux[:-1]) ** 2.) / ((len(flux) - 1) * np.var(flux))
            eta_arr.append(eta)
            obj_arr.append(obj)

        idx_arr = np.argsort(eta_arr)

        for i in range(N_samples):
            obj = obj_arr[idx_arr[i]]
            filename = '%s/field%06d_%i_lc.png' % (folder,
                                                   obj.fieldid,
                                                   obj.object_id)
            obj.plot_lightcurves(filename)

            source = create_source_from_object(obj)
            hmjd_g, mag_g, magerr_g = _return_obj_lightcurve_data(source.object_g)
            hmjd_r, mag_r, magerr_r = _return_obj_lightcurve_data(source.object_r)
            hmjd_i, mag_i, magerr_i = _return_obj_lightcurve_data(source.object_i)

            filename = filename.replace('.png', '.npz')
            np.savez(filename,
                     hmjd_g=hmjd_g, mag_g=mag_g, magerr_g=magerr_g,
                     hmjd_r=hmjd_r, mag_r=mag_r, magerr_r=magerr_r,
                     hmjd_i=hmjd_i, mag_i=mag_i, magerr_i=magerr_i)


def find_nonvariable_sample():
    folder = 'variable_sample/nonvariable'
    if not os.path.exists(folder):
        os.makedirs(folder)

    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.uploaded == True).all()
    job_ids = [j.id for j in jobs]
    random.shuffle(job_ids)

    samples = []
    for i, job_id in enumerate(job_ids):
        print(i, len(job_ids), len(samples))
        sources = db.session.query(Source).filter(Source.ingest_job_id == job_id).limit(100).all()
        print('-- %i sources' % len(sources))

        for source in sources:
            if source.zort_source.object_g is None:
                continue
            if source.zort_source.object_r is None:
                continue
            if source.zort_source.object_g.nepochs < 20:
                continue
            if source.zort_source.object_r.nepochs < 20:
                continue
            samples.append(source)
            break

        if len(samples) >= 20:
            break

    for sample in samples:
        obj = [o for o in sample.zort_source.objects if not None][0]
        filename = '%s/field%06d_%i_lc.png' % (folder, obj.fieldid, obj.object_id)
        obj.plot_lightcurves(filename)

        hmjd_g, mag_g, magerr_g = _return_obj_lightcurve_data(sample.zort_source.object_g)
        hmjd_r, mag_r, magerr_r = _return_obj_lightcurve_data(sample.zort_source.object_r)
        hmjd_i, mag_i, magerr_i = _return_obj_lightcurve_data(sample.zort_source.object_i)

        filename = filename.replace('.png', '.npz')
        np.savez(filename,
                 hmjd_g=hmjd_g, mag_g=mag_g, magerr_g=magerr_g,
                 hmjd_r=hmjd_r, mag_r=mag_r, magerr_r=magerr_r,
                 hmjd_i=hmjd_i, mag_i=mag_i, magerr_i=magerr_i)


def find_microlensing_sample():
    folder = 'variable_sample/microlensing'
    if not os.path.exists(folder):
        os.makedirs(folder)

    radec = np.array([[286.633211, 32.248996],
                      [290.784286, 7.810517],
                      [326.173116, 59.377872],
                      [339.955528, 51.647223],
                      [290.617225, 1.706486],
                      [284.029167, 13.152260],
                      [271.850400, -10.314477],
                      [271.439120, -12.014556],
                      [284.338291, 11.433438],
                      [285.984027, -13.929477],
                      [307.149376, 22.830478],
                      [287.113964, 1.531903],
                      [285.134471, 30.511120],
                      [279.578723, 7.837854],
                      [307.149376, 22.830478],
                      [291.019150, 20.478976],
                      [76.632447, 8.425664],
                      [48.694244, 62.343390],
                      [279.404621, 11.200516],
                      [55.197569, 57.955805],
                      [289.114418, 26.653532],
                      [280.734529, 32.873054],
                      [273.900566, -2.256985],
                      [274.913476, 0.590991],
                      [290.663294, 19.550373],
                      [258.208411, -27.182057],
                      [297.706148, 34.637344],
                      [281.836951, -4.338099],
                      [309.034132, 32.720880],
                      [283.497170, -1.152267]])
    for ra, dec in radec:
        sources = locate_sources_by_radec(ra, dec)
        for source in sources:
            if source.object_g:
                obj = source.object_g
            else:
                obj = source.object_r
            filename = '%s/field%06d_%i_lc.png' % (folder,
                                                   obj.fieldid,
                                                   obj.object_id)
            obj.plot_lightcurves(filename)

            hmjd_g, mag_g, magerr_g = _return_obj_lightcurve_data(source.object_g)
            hmjd_r, mag_r, magerr_r = _return_obj_lightcurve_data(source.object_r)
            hmjd_i, mag_i, magerr_i = _return_obj_lightcurve_data(source.object_i)

            filename = filename.replace('.png', '.npz')
            np.savez(filename,
                     hmjd_g=hmjd_g, mag_g=mag_g, magerr_g=magerr_g,
                     hmjd_r=hmjd_r, mag_r=mag_r, magerr_r=magerr_r,
                     hmjd_i=hmjd_i, mag_i=mag_i, magerr_i=magerr_i)
