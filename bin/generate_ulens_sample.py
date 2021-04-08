import os
import glob
import numpy as np
from collections import defaultdict
from scipy.stats import expon
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from zort.photometry import fluxes_to_magnitudes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle import db
from puzle.models import Source, SourceIngestJob
from puzle.utils import return_data_dir, save_stacked_array, return_DR4_dir, load_stacked_array
from puzle.stats import calculate_eta_on_daily_avg, calculate_eta_on_daily_avg_residuals

popsycle_base_folder = '/global/cfs/cdirs/uLens/PopSyCLE_runs/PopSyCLE_runs_v3_refined_events'


def gather_PopSyCLE_refined_events():
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
    fis = glob.glob(f'{popsycle_base_folder}/*fits')
    fis.sort()
    lb_arr = []
    for fi in fis:
        lb = os.path.basename(fi).split('_')
        l = float(lb[0].replace('l', ''))
        b = float(lb[1].replace('b', ''))
        lb_arr.append((l, b))
    return lb_arr


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


def fetch_objects(ra, dec, radius, limit, n_days_min=20):
    cone_filter = SourceIngestJob.cone_search(ra, dec, radius)
    jobs = db.session.query(SourceIngestJob).filter(cone_filter).all()

    n_samples_per_source = max(1, int(10 * (limit / len(jobs))))
    DR4_dir = return_DR4_dir()

    lightcurve_file_pointers = {}
    objects = []
    for i, job in enumerate(jobs):
        if i % 10 == 0:
            print('Processing job %i/%i | %i objects' % (i, len(jobs), len(objects)))
        source_job_id = job.id
        dir = '%s/sources_%s' % (DR4_dir, str(source_job_id)[:3])
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
            zort_source = source.load_zort_source()
            n_days_arr = []
            for obj in zort_source.objects:
                n_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
                n_days_arr.append(n_days)
            if np.max(n_days_arr) < n_days_min:
                continue
            obj = zort_source.objects[np.argmax(n_days_arr)]
            objects.append(obj)

        if len(objects) >= limit:
            break

    for file in lightcurve_file_pointers.values():
        file.close()

    return objects


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
                                   n_days_min=20):
    popsycle_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
    popsycle_catalog = Table.read(popsycle_fname, format='fits')

    tE_log_catalog = np.log10(popsycle_catalog['t_E'])
    tE_log_median = np.median(tE_log_catalog)
    tE_log_std = np.std(tE_log_catalog)
    tE_arr = 10 ** np.random.normal(tE_log_median, tE_log_std, size=N_samples*10)
    tE_arr_idx = np.where(tE_arr >= tE_min)[0]
    tE_arr_idx = np.random.choice(tE_arr_idx, size=N_samples, replace=False)
    tE_arr = tE_arr[tE_arr_idx]

    pi_E_catalog = popsycle_catalog['pi_E']
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
    metadata = []

    increment = N_samples / 10
    for i, obj in enumerate(objects):
        if i % increment == 0:
            print('Constructing object %i / %i' % (i, N_samples))
        tE = tE_arr[i]
        piE_E = piE_E_arr[i]
        piE_N = piE_N_arr[i]
        u0 = u0_arr[i]
        b_sff = b_sff_arr[i]
        mag_src = np.median(obj.lightcurve.mag) * b_sff

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
                    metadata.append((t0, u0, tE, mag_src, piE_E, piE_N, b_sff, obj.ra, obj.dec, eta_residual))
                    break

    return lightcurves, metadata


def generate_random_lightcurves():
    N_samples = 100
    tE_min = 20
    delta_m_min = 0.1
    delta_m_min_cut = 3
    n_days_min = 20
    lb_arr = gather_PopSyCLE_lb()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_lb_arr = np.array_split(lb_arr, size)[rank]

    lightcurves_arr = []
    metadata_arr = []
    for i, (l, b) in enumerate(my_lb_arr):
        print('Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (l, b, i, len(my_lb_arr)))
        lightcurves, metadata = generate_random_lightcurves_lb(l, b,
                                                               N_samples=N_samples, tE_min=tE_min,
                                                               delta_m_min=delta_m_min, delta_m_min_cut=delta_m_min_cut,
                                                               n_days_min=n_days_min)
        lightcurves_arr += lightcurves
        metadata_arr += metadata

    data_dir = return_data_dir()
    fname = f'{data_dir}/ulens_sample.{rank:02d}.npz'
    save_stacked_array(fname, lightcurves_arr)

    dtype = [('t0', float), ('u0', float),
             ('tE', float), ('mag_src', float),
             ('piE_E', float), ('piE_N', float),
             ('b_sff', float), ('ra', float),
             ('dec', float), ('eta_residual', float)]
    metadata_arr = np.array(metadata_arr, dtype=dtype)
    metadata_dct = {k: metadata_arr[k] for k in metadata_arr.dtype.names}

    fname = f'{data_dir}/ulens_sample_metadata.{rank:02d}.npz'
    np.savez(fname, **metadata_dct)


def consolidate_lightcurves():
    data_dir = return_data_dir()
    ulens_sample_fnames = glob.glob(f'{data_dir}/ulens_sample.??.npz')
    ulens_sample_fnames.sort()

    lightcurves_arr = []
    metadata_dct = defaultdict(list)
    for fname in ulens_sample_fnames:
        data = load_stacked_array(fname)
        for d in data:
            lightcurves_arr.append((d[:,0], d[:,1], d[:,2]))

        fname_metadata = fname.replace('sample.', 'sample_metadata.')
        metadata = np.load(fname_metadata)
        for key in metadata:
            metadata_dct[key].extend(list(metadata[key]))

    fname_total = f'{data_dir}.ulens_sample.total.npz'
    save_stacked_array(fname_total, lightcurves_arr)

    fname_total = f'{data_dir}/ulens_sample_metadata.total.npz'
    np.savez(fname_total, **metadata_dct)


if __name__ == '__main__':
    generate_random_lightcurves()
