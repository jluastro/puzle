#! /usr/bin/env python
"""
generate_ulens_sample.py
"""

import os
import glob
import numpy as np
from collections import defaultdict
from scipy.stats import expon
from sqlalchemy.sql.expression import func
import itertools

from zort.photometry import fluxes_to_magnitudes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle import db
from puzle.ulens import return_ulens_data_fname
from puzle.cands import fit_data_to_ulens_opt, calculate_chi2_model
from puzle.models import Source, SourceIngestJob
from puzle.utils import return_data_dir, save_stacked_array, \
    return_DR5_dir, load_stacked_array, sortsplit
from puzle.stats import calculate_eta_on_daily_avg, \
    calculate_eta_on_daily_avg_residuals, average_xy_on_round_x

popsycle_base_folder = '/global/cfs/cdirs/uLens/PopSyCLE_runs/PopSyCLE_runs_v3_refined_events'


def gather_PopSyCLE_refined_events():
    from astropy.table import Table, vstack

    folders = glob.glob('PopSyCLE_runs_v3/l*')
    folders.sort()
    N_folders = len(folders)
    N_samples = 0
    for i, folder in enumerate(folders):
        print(f'Processing {folder} ({i}/{N_folders})')
        lb = os.path.basename(folder)
        fis = glob.glob(f'{folder}/*_5yrs_refined_events_ztf_r_Damineli16.fits')
        print('-- %i files' % len(fis))

        tables = []
        for fi in fis:
            t = Table.read(fi, format='fits')
            N_samples += len(t)
            tables.append(t)
        table_new = vstack(tables)

        fi_new = f'{popsycle_base_folder}/{lb}_refined_events_ztf_r_Damineli16.fits'
        table_new.write(fi_new, overwrite=True)

    print(f'{N_samples} Samples')


def gather_PopSyCLE_lb():
    fis = glob.glob(f'{popsycle_base_folder}/*ztf_r*fits')
    fis.sort()
    lb_skip_arr = [(2.0, -1.0), (2.0, 1.0), (6.0, -1.0),
                   (6.0, -3.0), (6.0, -6.0), (6.0, 1.0)]
    lb_arr = []
    for fi in fis:
        lb = os.path.basename(fi).split('_')
        l = float(lb[0].replace('l', ''))
        b = float(lb[1].replace('b', ''))
        if (l, b) in lb_skip_arr:
            continue
        lb_arr.append((l, b))
    return np.array(lb_arr)


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


def fetch_objects(ra, dec, radius, limit, n_days_min=50):
    cone_filter = SourceIngestJob.cone_search(ra, dec, radius)
    jobs = db.session.query(SourceIngestJob).filter(cone_filter).order_by(func.random()).all()

    n_samples_per_source = max(1, int(5 * (limit / len(jobs))))
    DR5_dir = return_DR5_dir()

    lightcurve_file_pointers = {}
    objects = []
    for i, job in enumerate(jobs):
        if i % 10 == 0:
            print('Fetching job %i/%i | %i objects' % (i, len(jobs), len(objects)))
        source_job_id = job.id
        dir = '%s/sources_%s' % (DR5_dir, str(source_job_id)[:3])
        fname = f'{dir}/sources.{source_job_id:06}.txt'
        if not os.path.exists(fname):
            continue

        lines = open(fname, 'r').readlines()[1:]
        if len(lines) == 0:
            continue

        size = min(len(lines), n_samples_per_source)
        idx_arr = np.random.choice(np.arange(len(lines)),
                                   size=size,
                                   replace=False)
        for idx in idx_arr:
            source = csv_line_to_source(lines[idx], lightcurve_file_pointers)
            if source.object_id_g is None or source.object_id_r is None:
                continue
            zort_source = source.load_zort_source()
            n_days_arr = []
            for obj in zort_source.objects:
                n_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
                n_days_arr.append(n_days)
            if (len(n_days_arr) == 0) or (np.max(n_days_arr) < n_days_min):
                continue
            obj = zort_source.objects[np.argmax(n_days_arr)]
            if obj.color == 'r':
                sib = zort_source.object_g
            else:
                sib = zort_source.object_r
            n_days = len(np.unique(np.round(sib.lightcurve.hmjd)))
            if n_days < n_days_min:
                continue
            obj.siblings = [sib]
            objects.append(obj)

        if len(objects) >= limit:
            break

    for file in lightcurve_file_pointers.values():
        file.close()

    return objects[:limit]


def calculate_delta_m(u0, b_sff):
    # delta_f = (a * f_S + f_LN) / (f_T)
    # delta_f = (a * b_sff * f_T + (1 - b_sff) * f_T) / (f_T)
    # delta_f = a * b_sff + (1 - b_sff) = 1 + (a - 1) * b_sff
    # delta_m = | -2.5 * np.log10(delta_f) |

    amp = np.abs((u0 ** 2 + 2) / (u0 * np.sqrt(u0 ** 2 + 4)))
    delta_f = 1 + (amp - 1) * b_sff
    delta_m = 2.5 * np.log10(delta_f)
    return delta_m


def generate_random_lightcurves_lb(l, b, N_samples=1000,
                                   tE_min=20, delta_m_min=0.1, delta_m_min_cut=3,
                                   n_days_min=50):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    popsycle_r_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
    popsycle_r_catalog = Table.read(popsycle_r_fname, format='fits')
    # popsycle_g_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_g_Damineli16.fits'
    # popsycle_g_catalog = Table.read(popsycle_g_fname, format='fits')

    tE_log_catalog = np.log10(popsycle_r_catalog['t_E'])
    tE_log_median = np.median(tE_log_catalog)
    tE_log_std = np.std(tE_log_catalog)
    tE_arr = 10 ** np.random.normal(tE_log_median, tE_log_std, size=N_samples*10)
    tE_arr_idx = np.where(tE_arr >= tE_min)[0]
    tE_arr_idx = np.random.choice(tE_arr_idx, size=N_samples, replace=False)
    tE_arr = tE_arr[tE_arr_idx]

    pi_E_catalog = popsycle_r_catalog['pi_E']
    loc, scale = expon.fit(pi_E_catalog)
    pi_E = expon.rvs(loc, scale, N_samples)
    theta = np.random.uniform(0, 2 * np.pi, N_samples)
    piE_E_arr = pi_E * np.cos(theta)
    piE_N_arr = pi_E * np.sin(theta)

    u0_arr = np.random.uniform(-2, 2, N_samples * 10)
    b_sff_arr = np.random.uniform(0, 1, N_samples * 10)
    delta_m = calculate_delta_m(u0_arr, b_sff_arr)
    delta_m_idx_arr = np.where(delta_m >= delta_m_min)[0]
    delta_m_idx_arr = np.random.choice(delta_m_idx_arr, size=N_samples, replace=False)
    u0_arr = u0_arr[delta_m_idx_arr]
    b_sff_arr = b_sff_arr[delta_m_idx_arr]

    coord = SkyCoord(l, b, unit=u.degree, frame='galactic')
    ra, dec = coord.icrs.ra.value, coord.icrs.dec.value
    radius = np.sqrt(47 / np.pi) * 3600.

    objects = fetch_objects(ra, dec, radius,
                            limit=N_samples, n_days_min=n_days_min)
    lightcurves = []
    lightcurves_sibs = []
    metadata = []
    metadata_sibs = []

    increment = N_samples / 10
    for i, obj in enumerate(objects):
        if i % increment == 0:
            print('Constructing object %i / %i' % (i, N_samples))
        tE = tE_arr[i]
        piE_E = piE_E_arr[i]
        piE_N = piE_N_arr[i]
        u0 = u0_arr[i]
        b_sff = b_sff_arr[i]
        flux_src = np.median(obj.lightcurve.flux) * b_sff
        mag_src, _ = fluxes_to_magnitudes(flux_src)

        obj_t = obj.lightcurve.hmjd
        obj_mag = obj.lightcurve.mag
        obj_magerr = obj.lightcurve.magerr
        obj_flux = obj.lightcurve.flux
        obj_fluxerr = obj.lightcurve.fluxerr
        try:
            t0_min, t0_max = np.min(obj_t), np.max(obj_t)
        except ValueError:
            print('Exiting sample %i due to ValueError on obj_t' % i)
            continue

        attempts = 0
        while True:
            attempts += 1
            if attempts >= N_samples:
                print('Exiting sample %i due to excessive attempts at t0' % i)
                break
            t0 = np.random.uniform(t0_min, t0_max)
            model = PSPL_Phot_Par_Param1(t0=t0, u0_amp=u0, tE=tE, mag_src=mag_src,
                                         piE_E=piE_E, piE_N=piE_N, b_sff=b_sff,
                                         raL=obj.ra, decL=obj.dec)
            # f_micro = A * f_source + f_neighbor_lens
            # f_neighbor_lens = f_total - f_source = f_total - b_sff * f_total = (1 - b_sff) * f_total
            # f_micro = A * b_sff * f_total + (1 - b_sff) * f_total

            amp = model.get_amplification(obj_t)
            obj_flux_micro = amp * b_sff * obj_flux + (1 - b_sff) * obj_flux
            obj_mag_micro, _ = fluxes_to_magnitudes(obj_flux_micro, obj_fluxerr)
            delta_m = obj_mag - obj_mag_micro
            delta_m_min_cond = delta_m >= delta_m_min

            eta_residual = calculate_eta_on_daily_avg(obj_t, obj_mag)
            if np.sum(delta_m_min_cond) >= delta_m_min_cut:
                eta_residual_daily = calculate_eta_on_daily_avg_residuals(obj_t, obj_mag_micro, obj_magerr)
                if eta_residual_daily is not None and not np.isnan(eta_residual_daily):
                    lightcurves.append((obj_t, obj_mag_micro, obj_magerr))
                    metadata.append((obj.filename, obj.object_id, obj.lightcurve_position,
                                     t0, u0, tE, mag_src, piE_E, piE_N, b_sff, obj.ra, obj.dec, eta_residual))

                    # append sibling data
                    sib = obj.siblings[0]
                    obj_sib_t = sib.lightcurve.hmjd
                    obj_sib_mag = sib.lightcurve.mag
                    obj_sib_magerr = sib.lightcurve.magerr
                    obj_sib_flux = sib.lightcurve.flux
                    obj_sib_fluxerr = sib.lightcurve.fluxerr

                    flux_src_sib = np.median(obj_sib_flux) * b_sff
                    mag_src_sib, _ = fluxes_to_magnitudes(flux_src_sib)
                    model_sib = PSPL_Phot_Par_Param1(t0=t0, u0_amp=u0, tE=tE, mag_src=mag_src_sib,
                                                     piE_E=piE_E, piE_N=piE_N, b_sff=b_sff,
                                                     raL=sib.ra, decL=sib.dec)

                    amp_sib = model_sib.get_amplification(obj_sib_t)
                    obj_sib_flux_micro = amp_sib * b_sff * obj_sib_flux + (1 - b_sff) * obj_sib_flux
                    obj_sib_mag_micro, _ = fluxes_to_magnitudes(obj_sib_flux_micro, obj_sib_fluxerr)
                    eta_residual_sib = calculate_eta_on_daily_avg(obj_sib_t, obj_sib_mag)

                    lightcurves_sibs.append((obj_sib_t, obj_sib_mag_micro, obj_sib_magerr))
                    metadata_sibs.append((sib.filename, sib.object_id, sib.lightcurve_position,
                                          t0, u0, tE, mag_src, piE_E, piE_N, b_sff, sib.ra, sib.dec, eta_residual_sib))
                    break

    return lightcurves, metadata, lightcurves_sibs, metadata_sibs


def generate_random_lightcurves():
    N_samples = 2000
    tE_min = 20
    delta_m_min = 0.1
    delta_m_min_cut = 3
    n_days_min = 50
    lb_arr = gather_PopSyCLE_lb()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_lb_arr = sortsplit(lb_arr, size)[rank]

    lightcurves_arr = []
    metadata_arr = []
    lightcurves_sibs_arr = []
    metadata_sibs_arr = []
    for i, (l, b) in enumerate(my_lb_arr):
        print('%i) Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (rank, l, b, i, len(my_lb_arr)))
        results = generate_random_lightcurves_lb(l, b,
                                                 N_samples=N_samples, tE_min=tE_min,
                                                 delta_m_min=delta_m_min, delta_m_min_cut=delta_m_min_cut,
                                                 n_days_min=n_days_min)
        lightcurves, metadata, lightcurves_sibs, metadata_sibs = results
        print('%i) Processing (l, b) = (%.2f, %.2f) |  %i lightcurves' % (rank, l, b, len(lightcurves)))
        lightcurves_arr += lightcurves
        metadata_arr += metadata
        lightcurves_sibs_arr += lightcurves_sibs
        metadata_sibs_arr += metadata_sibs

    if len(lightcurves_arr) == 0:
        print('%i) No lightcurves generated!' % rank)
        return
    else:
        print('%i) %i lightcurves generated' % (rank, len(lightcurves_arr)))


    data_dir = return_data_dir()
    fname = f'{data_dir}/ulens_samples/ulens_sample.{rank:02d}.npz'
    if os.path.exists(fname):
        os.remove(fname)
    save_stacked_array(fname, lightcurves_arr)

    fname = f'{data_dir}/ulens_samples/ulens_sample.sibs.{rank:02d}.npz'
    if os.path.exists(fname):
        os.remove(fname)
    save_stacked_array(fname, lightcurves_sibs_arr)

    dtype = [('filename', str), ('id', int), ('lightcurve_position', int),
             ('t0', float), ('u0', float),
             ('tE', float), ('mag_src', float),
             ('piE_E', float), ('piE_N', float),
             ('b_sff', float), ('ra', float),
             ('dec', float), ('eta_residual', float)]
    metadata_arr = np.array(metadata_arr, dtype=dtype)
    metadata_dct = {k: metadata_arr[k] for k in metadata_arr.dtype.names}
    metadata_sibs_arr = np.array(metadata_sibs_arr, dtype=dtype)
    metadata_sibs_dct = {k: metadata_sibs_arr[k] for k in metadata_sibs_arr.dtype.names}

    fname = f'{data_dir}/ulens_samples/ulens_sample_metadata.{rank:02d}.npz'
    if os.path.exists(fname):
        os.remove(fname)
    np.savez(fname, **metadata_dct)

    fname = f'{data_dir}/ulens_samples/ulens_sample_metadata.sibs.{rank:02d}.npz'
    if os.path.exists(fname):
        os.remove(fname)
    np.savez(fname, **metadata_sibs_dct)

    if 'SLURMD_NODENAME' in os.environ:
        comm.Barrier()


def consolidate_lightcurves():
    # run generate_random_lightcurves first
    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    if rank == 0:

        # get list of already existing totals
        data_dir = return_data_dir()
        fname_total_arr = glob.glob(f'{data_dir}/ulens_sample.??.total.npz')
        fname_total_arr.sort()
        if len(fname_total_arr) == 0:
            idx = 0
        else:
            idx = int(os.path.basename(fname_total_arr[-1]).split('.')[1]) + 1

        # load existing totals into memory
        lightcurves_arr = []
        metadata_dct = defaultdict(list)
        lightcurves_sibs_arr = []
        metadata_sibs_dct = defaultdict(list)
        for fname in fname_total_arr:
            data = load_stacked_array(fname)
            for d in data:
                lightcurves_arr.append((d[:, 0], d[:, 1], d[:, 2]))

            data = load_stacked_array(fname.replace('sample.', 'sample.sibs.'))
            for d in data:
                lightcurves_sibs_arr.append((d[:, 0], d[:, 1], d[:, 2]))

            fname_metadata = fname.replace('sample.', 'sample_metadata.')
            metadata = np.load(fname_metadata)
            for key in metadata:
                metadata_dct[key].extend(list(metadata[key]))

            fname_metadata = fname.replace('sample.', 'sample_metadata.sibs.')
            metadata_sibs = np.load(fname_metadata)
            for key in metadata:
                metadata_sibs_dct[key].extend(list(metadata_sibs[key]))

        # append with new samples
        ulens_sample_fnames = glob.glob(f'{data_dir}/ulens_samples/ulens_sample.??.npz')
        ulens_sample_fnames.sort()

        for fname in ulens_sample_fnames:
            data = load_stacked_array(fname)
            for d in data:
                lightcurves_arr.append((d[:, 0], d[:, 1], d[:, 2]))

            data = load_stacked_array(fname.replace('sample.', 'sample.sibs.'))
            for d in data:
                lightcurves_sibs_arr.append((d[:, 0], d[:, 1], d[:, 2]))

            fname_metadata = fname.replace('sample.', 'sample_metadata.')
            metadata = np.load(fname_metadata)
            for key in metadata:
                metadata_dct[key].extend(list(metadata[key]))

            fname_metadata = fname.replace('sample.', 'sample_metadata.sibs.')
            metadata_sibs = np.load(fname_metadata)
            for key in metadata:
                metadata_sibs_dct[key].extend(list(metadata_sibs[key]))

        fname_total = f'{data_dir}/ulens_sample.{idx:02d}.total.npz'
        save_stacked_array(fname_total, lightcurves_arr)

        fname_total = f'{data_dir}/ulens_sample.sibs.{idx:02d}.total.npz'
        save_stacked_array(fname_total, lightcurves_sibs_arr)

        fname_metadata_total = f'{data_dir}/ulens_sample_metadata.{idx:02d}.total.npz'
        np.savez(fname_metadata_total, **metadata_dct)

        fname_metadata_total = f'{data_dir}/ulens_sample_metadata.sibs.{idx:02d}.total.npz'
        np.savez(fname_metadata_total, **metadata_sibs_dct)

        num_samples = len(lightcurves_arr)
        print(f'{num_samples} Samples in Total')

    if 'SLURMD_NODENAME' in os.environ:
        comm.Barrier()


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


def three_consecutive_decreases(arr):
    return np.array([strictly_decreasing(arr[i:i+3]) for i in range(len(arr)-2)])


def test_for_three_consecutive_decreases(arr):
    return any(three_consecutive_decreases(arr))


def _calculate_stats_on_lightcurves(sibsFlag=False):
    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    # run consolidate lightcurves first
    data_dir = return_data_dir()
    my_stats_complete_fname = f'{data_dir}/ulens_samples/stats.{rank:02d}.txt'

    fname = return_ulens_data_fname('ulens_sample')
    if sibsFlag:
        fname = fname.replace('sample', 'sample.sibs')
    data = load_stacked_array(fname)
    lightcurve_data = []
    for i, d in enumerate(data):
        lightcurve_data.append(d)

    fname = return_ulens_data_fname('ulens_sample_metadata')
    if sibsFlag:
        fname = fname.replace('metadata', 'metadata.sibs')
    metadata = np.load(fname)

    idx_arr = np.arange(len(lightcurve_data))
    my_idx_arr = np.array_split(idx_arr, size)[rank]
    my_data = np.array_split(data, size)[rank]

    print('Rank %i) Processing %i lightcurves' % (rank, len(my_data)))

    param_names = ['t0', 'u0_amp', 'tE', 'mag_src',
                   'b_sff', 'piE_E', 'piE_N']
    model_class = PSPL_Phot_Par_Param1
    my_chi_squared_modeled_arr = []
    my_num_days_arr = []
    my_eta_arr = []
    my_eta_residual_level2_arr = []
    my_t0_level2_arr = []
    my_tE_level2_arr = []
    my_f0_level2_arr = []
    my_f1_level2_arr = []
    my_chi_squared_delta_level2_arr = []
    my_chi_squared_flat_level2_arr = []
    my_atype_level2_arr = []
    my_t0_level3_arr = []
    my_u0_amp_level3_arr = []
    my_tE_level3_arr = []
    my_mag_src_level3_arr = []
    my_b_sff_level3_arr = []
    my_piE_E_level3_arr = []
    my_piE_N_level3_arr = []
    my_chi_squared_ulens_level3_arr = []
    my_eta_residual_level3_arr = []
    my_observable_arr1 = []
    my_observable_arr2 = []
    my_observable_arr3 = []
    for i, d in enumerate(my_data):
        hmjd = d[:, 0]
        mag = d[:, 1]
        magerr = d[:, 2]
        idx = my_idx_arr[i]

        # calculated chi2 modeled
        t0 = metadata['t0'][idx]
        u0 = metadata['u0'][idx]
        tE = metadata['tE'][idx]
        mag_src = metadata['mag_src'][idx]
        piE_E = metadata['piE_E'][idx]
        piE_N = metadata['piE_N'][idx]
        b_sff = metadata['b_sff'][idx]
        ra = metadata['ra'][idx]
        dec = metadata['dec'][idx]

        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        _, magerr_round = average_xy_on_round_x(hmjd, mag)

        data_fit = {'hmjd': hmjd_round, 'mag': mag_round, 'magerr': magerr_round, 'raL': ra, 'decL': dec}
        param_values = [t0, u0, tE, mag_src, b_sff, piE_E, piE_N]
        chi2 = calculate_chi2_model(param_values, param_names, model_class, data_fit)
        my_chi_squared_modeled_arr.append(chi2)
        my_num_days_arr.append(len(hmjd_round))

        # calculate and append eta and level2 fit data
        eta_daily = calculate_eta_on_daily_avg(hmjd, mag)
        my_eta_arr.append(eta_daily)

        eta_residual_daily, fit_data = calculate_eta_on_daily_avg_residuals(hmjd, mag, magerr,
                                                                            return_fit_data=True)
        if fit_data is not None:
            t0, tE, f0, f1, chi_squared_delta, chi_squared_flat, atype = fit_data
        else:
            t0, tE, f0, f1, chi_squared_delta, chi_squared_flat, atype = [0 for _ in range(7)]
            eta_residual_daily = 0

        my_eta_residual_level2_arr.append(eta_residual_daily)
        my_t0_level2_arr.append(t0)
        my_tE_level2_arr.append(tE)
        my_f0_level2_arr.append(f0)
        my_f1_level2_arr.append(f1)
        my_chi_squared_delta_level2_arr.append(chi_squared_delta)
        my_chi_squared_flat_level2_arr.append(chi_squared_flat)
        my_atype_level2_arr.append(atype)

        # calculate and append observability conditions
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        n_days_in_split = int(len(mag_round) / 5)
        mag_splits = [mag_round[i * n_days_in_split:(i + 1) * n_days_in_split] for i in range(5)]
        median = np.median([np.median(mag) for mag in mag_splits])
        std = np.median([np.std(mag) for mag in mag_splits])
        cond_three_sigma = mag_round <= median - 3 * std
        cond_decreasing = three_consecutive_decreases(mag_round)
        count_cond = np.sum(cond_three_sigma[:-2] * cond_decreasing)

        if count_cond == 0:
            my_observable_arr1.append(False)
            my_observable_arr2.append(False)
            my_observable_arr3.append(False)
        elif count_cond == 1:
            my_observable_arr1.append(True)
            my_observable_arr2.append(False)
            my_observable_arr3.append(False)
        elif count_cond == 2:
            my_observable_arr1.append(True)
            my_observable_arr2.append(True)
            my_observable_arr3.append(False)
        else:
            my_observable_arr1.append(True)
            my_observable_arr2.append(True)
            my_observable_arr3.append(True)

        # calculate and append level3 fit data
        if count_cond >= 3:
            if t0 == 0:
                t0_guess = None
                tE_guess = None
            else:
                t0_guess = t0
                tE_guess = tE
            best_params = fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec,
                                                t0_guess=t0_guess,
                                                tE_guess=tE_guess)
        else:
            best_params = {'t0': 0,
                           'u0_amp': 0,
                           'tE': 0,
                           'mag_src': 0,
                           'b_sff': 0,
                           'piE_E': 0,
                           'piE_N': 0,
                           'chi_squared_ulens': 0,
                           'eta_residual': 0}
        my_t0_level3_arr.append(best_params['t0'])
        my_u0_amp_level3_arr.append(best_params['u0_amp'])
        my_tE_level3_arr.append(best_params['tE'])
        my_mag_src_level3_arr.append(best_params['mag_src'])
        my_b_sff_level3_arr.append(best_params['b_sff'])
        my_piE_E_level3_arr.append(best_params['piE_E'])
        my_piE_N_level3_arr.append(best_params['piE_N'])
        my_chi_squared_ulens_level3_arr.append(best_params['chi_squared_ulens'])
        my_eta_residual_level3_arr.append(best_params['eta_residual'])

        with open(my_stats_complete_fname, 'a+') as f:
            f.write(f'{i}\n')

    total_chi_squared_modeled_arr = comm.gather(my_chi_squared_modeled_arr, root=0)
    total_num_days_arr = comm.gather(my_num_days_arr, root=0)
    total_eta_arr = comm.gather(my_eta_arr, root=0)
    total_eta_residual_level2_arr = comm.gather(my_eta_residual_level2_arr, root=0)
    total_observable_arr1 = comm.gather(my_observable_arr1, root=0)
    total_observable_arr2 = comm.gather(my_observable_arr2, root=0)
    total_observable_arr3 = comm.gather(my_observable_arr3, root=0)
    total_t0_level2_arr = comm.gather(my_t0_level2_arr, root=0)
    total_tE_level2_arr = comm.gather(my_tE_level2_arr, root=0)
    total_f0_level2_arr = comm.gather(my_f0_level2_arr, root=0)
    total_f1_level2_arr = comm.gather(my_f1_level2_arr, root=0)
    total_chi_squared_delta_level2_arr = comm.gather(my_chi_squared_delta_level2_arr, root=0)
    total_chi_squared_flat_level2_arr = comm.gather(my_chi_squared_flat_level2_arr, root=0)
    total_atype_level2_arr = comm.gather(my_atype_level2_arr, root=0)
    total_t0_level3_arr = comm.gather(my_t0_level3_arr, root=0)
    total_u0_amp_level3_arr = comm.gather(my_u0_amp_level3_arr, root=0)
    total_tE_level3_arr = comm.gather(my_tE_level3_arr, root=0)
    total_mag_src_level3_arr = comm.gather(my_mag_src_level3_arr, root=0)
    total_b_sff_level3_arr = comm.gather(my_b_sff_level3_arr, root=0)
    total_piE_E_level3_arr = comm.gather(my_piE_E_level3_arr, root=0)
    total_piE_N_level3_arr = comm.gather(my_piE_N_level3_arr, root=0)
    total_chi_squared_ulens_level3_arr = comm.gather(my_chi_squared_ulens_level3_arr, root=0)
    total_eta_residual_level3_arr = comm.gather(my_eta_residual_level3_arr, root=0)
    total_idx_arr = comm.gather(my_idx_arr, root=0)

    if rank == 0:
        chi_squared_modeled_arr = list(itertools.chain(*total_chi_squared_modeled_arr))
        num_days_arr = list(itertools.chain(*total_num_days_arr))
        eta_arr = list(itertools.chain(*total_eta_arr))
        eta_residual_level2_arr = list(itertools.chain(*total_eta_residual_level2_arr))
        observable_arr1 = list(itertools.chain(*total_observable_arr1))
        observable_arr2 = list(itertools.chain(*total_observable_arr2))
        observable_arr3 = list(itertools.chain(*total_observable_arr3))
        t0_level2_arr = list(itertools.chain(*total_t0_level2_arr))
        tE_level2_arr = list(itertools.chain(*total_tE_level2_arr))
        f0_level2_arr = list(itertools.chain(*total_f0_level2_arr))
        f1_level2_arr = list(itertools.chain(*total_f1_level2_arr))
        chi_squared_delta_level2_arr = list(itertools.chain(*total_chi_squared_delta_level2_arr))
        chi_squared_flat_level2_arr = list(itertools.chain(*total_chi_squared_flat_level2_arr))
        atype_level2_arr = list(itertools.chain(*total_atype_level2_arr))
        t0_level3_arr = list(itertools.chain(*total_t0_level3_arr))
        u0_amp_level3_arr = list(itertools.chain(*total_u0_amp_level3_arr))
        tE_level3_arr = list(itertools.chain(*total_tE_level3_arr))
        mag_src_level3_arr = list(itertools.chain(*total_mag_src_level3_arr))
        b_sff_level3_arr = list(itertools.chain(*total_b_sff_level3_arr))
        piE_E_level3_arr = list(itertools.chain(*total_piE_E_level3_arr))
        piE_N_level3_arr = list(itertools.chain(*total_piE_N_level3_arr))
        chi_squared_ulens_level3_arr = list(itertools.chain(*total_chi_squared_ulens_level3_arr))
        eta_residual_level3_arr = list(itertools.chain(*total_eta_residual_level3_arr))
        idx_arr = list(itertools.chain(*total_idx_arr))
        fname_data = return_ulens_data_fname('ulens_sample')
        fname_stats = fname_data.replace('ulens_sample', 'ulens_sample_stats')
        if sibsFlag:
            fname_stats = fname_stats.replace('stats', 'stats.sibs')
        np.savez(fname_stats,
                 chi_squared_modeled=chi_squared_modeled_arr,
                 num_days=num_days_arr,
                 eta=eta_arr,
                 eta_residual_level2=eta_residual_level2_arr,
                 observable1=observable_arr1,
                 observable2=observable_arr2,
                 observable3=observable_arr3,
                 t0_level2=t0_level2_arr,
                 tE_level2=tE_level2_arr,
                 f0_level2=f0_level2_arr,
                 f1_level2=f1_level2_arr,
                 chi_squared_delta_level2=chi_squared_delta_level2_arr,
                 chi_squared_flat_level2=chi_squared_flat_level2_arr,
                 atype_level2=atype_level2_arr,
                 t0_level3=t0_level3_arr,
                 u0_amp_level3=u0_amp_level3_arr,
                 tE_level3=tE_level3_arr,
                 mag_src_level3=mag_src_level3_arr,
                 b_sff_level3=b_sff_level3_arr,
                 piE_E_level3=piE_E_level3_arr,
                 piE_N_level3=piE_N_level3_arr,
                 chi_squared_ulens_level3=chi_squared_ulens_level3_arr,
                 eta_residual_level3=eta_residual_level3_arr,
                 idx=idx_arr)


def calculate_stats_on_lightcurves():
    # run consolidate lightcurves first

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        rank = 0

    data_dir = return_data_dir()
    my_stats_complete_fname = f'{data_dir}/ulens_samples/stats.{rank:02d}.txt'
    if os.path.exists(my_stats_complete_fname):
        os.remove(my_stats_complete_fname)

    _calculate_stats_on_lightcurves(sibsFlag=False)
    _calculate_stats_on_lightcurves(sibsFlag=True)


def _test_lightcurve_stats(data_lightcurves, observable_arr, idx_sample):
    print('-- %i lightcurves' % len(idx_sample))
    n_failures = 0
    for idx in idx_sample:
        d = data_lightcurves[idx]
        hmjd = d[:, 0]
        mag = d[:, 1]
        obs = observable_arr[idx]

        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        n_days_in_split = int(len(mag_round) / 5)
        mag_splits = [mag_round[i * n_days_in_split:(i + 1) * n_days_in_split] for i in range(5)]
        median = np.median([np.median(mag) for mag in mag_splits])
        std = np.median([np.std(mag) for mag in mag_splits])

        cond_three_sigma = mag_round <= median - 3 * std
        cond_descreasing = three_consecutive_decreases(mag_round)
        count_cond = np.sum(cond_three_sigma[:-2] * cond_descreasing)

        try:
            assert (count_cond >= 3) == obs
        except AssertionError:
            n_failures += 1
    print('-- %i Failures' % n_failures)


def test_lightcurve_stats(N_samples=1000):

    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    data_lightcurves = load_stacked_array(fname)

    fname_etas = fname.replace('ulens_sample', 'ulens_sample_stats')
    data_etas = np.load(fname_etas)
    observable_arr = data_etas['observable3']
    assert np.all(data_etas['idx_check'] == np.arange(len(observable_arr)))

    print('Testing all lightcurves')
    idx_sample = np.random.choice(np.arange(len(data_lightcurves)),
                                  size=N_samples, replace=False)
    _test_lightcurve_stats(data_lightcurves, observable_arr, idx_sample)

    print('Testing observable lightcurves')
    idx_observable = np.where(observable_arr == True)[0]
    idx_sample_obs = np.random.choice(idx_observable,
                                      size=N_samples, replace=False)
    _test_lightcurve_stats(data_lightcurves, observable_arr, idx_sample_obs)


if __name__ == '__main__':
    # generate_random_lightcurves()
    # consolidate_lightcurves()
    calculate_stats_on_lightcurves()
