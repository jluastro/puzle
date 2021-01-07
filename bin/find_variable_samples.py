#! /usr/bin/env python
"""
calculate_obj_snr_cuts.py
"""
import glob
import numpy as np
from zort.lightcurveFile import LightcurveFile


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def find_variable_samples(field_id, N_candidates=200000, N_samples=5):
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
        for t in np.arange(hmjd_min, hmjd_max+1):
            cond = np.floor(hmjd) == t
            if np.sum(cond) == 0:
                continue
            flux.append(np.median(obj.lightcurve.flux[cond]))
            t_arr.append(t)
        flux = np.array(flux)

        eta = np.sum((flux[:-1] - flux[1:]) ** 2.) / ((len(flux) - 1) * np.var(flux))
        eta_arr.append(eta)
        obj_arr.append(obj)

    idx_arr = np.argsort(eta_arr)

    for i in range(N_samples):
        obj = obj_arr[idx_arr[i]]
        obj.locate_siblings()
        obj.plot_lightcurves()

        fname = filename.replace('.txt', '_') + 'sample%02d.npz' % i
        np.savez(fname,
                 mag=obj.lightcurve.mag,
                 magerr=obj.lightcurve.magerr,
                 hmjd=obj.lightcurve.hmjd)


if __name__ == '__main__':
    field_id_arr = [282, 283, 333, 334, 539, 590]
    for field_id in field_id_arr:
        find_variable_samples(field_id)
