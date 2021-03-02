#! /usr/bin/env python
"""
sample.py
"""

import os
import glob
import numpy as np
from scipy.stats import expon
from astropy.table import Table, vstack
from shapely.geometry.polygon import Polygon

from zort.radec import return_ZTF_RCID_corners
from zort.photometry import fluxes_to_magnitudes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle import catalog
from puzle.utils import lightcurve_file_to_field_id, popsycle_base_folder
from puzle.models import Source
from puzle import db


def gather_PopSyCLE_refined_events():
    folders = glob.glob('PopSyCLE_runs_v3/l*')
    folders.sort()
    N_folders = len(folders)
    N_samples = 0
    for i, folder in enumerate(folders):
        print(f'Processing {folder} ({i}/{N_folders})')
        lb = os.path.basename(folder)
        fis = glob.glob(f'{folder}/*_5yrs_refined_events_ztf_r_Damineli16.fits')

        tables = []
        for fi in fis:
            t = Table.read(fi, format='fits')
            N_samples += len(t)
            tables.append(t)
        table_new = vstack(tables)

        fi_new = f'{popsycle_base_folder}/{lb}_refined_events_ztf_r_Damineli16.fits'
        table_new.write(fi_new, overwrite=True)

    print(f'{N_samples} Samples')


def fetch_sample_objects(lightcurve_file, n_days_min=20, rf_threshold=0.83,
                         num_sources_in_query=150, rcid_radius=1250):
    ulens_con = catalog.ulens_con()

    field_id = lightcurve_file_to_field_id(lightcurve_file)
    ZTF_RCID_corners = return_ZTF_RCID_corners(field_id)

    objs = []
    for rcid, corners in ZTF_RCID_corners.items():
        print(rcid)
        polygon = Polygon(corners)
        ra_rcid, dec_rcid = polygon.centroid.x, polygon.centroid.y

        query = db.session.query(Source).\
            filter(Source.cone_search(ra_rcid, dec_rcid, rcid_radius)).\
            order_by(Source.id)
        num_sources_db = query.count()

        num_objs = 0
        for row_min in range(0, num_sources_db, num_sources_in_query):
            sources_db = query.offset(row_min).limit(num_sources_in_query).all()
            for source in sources_db:
                zort_source = source.load_zort_source()
                nepochs_arr = [obj.nepochs for obj in zort_source.objects]
                obj = zort_source.objects[np.argmax(nepochs_arr)]
                n_days = len(np.unique(np.round(obj.lightcurve.hmjd)))
                ps1_psc = catalog.query_ps1_psc(obj.ra, obj.dec, con=ulens_con)
                if ps1_psc is None:
                    continue
                if n_days >= n_days_min and ps1_psc.rf_score >= rf_threshold:
                    objs.append(obj)
                    num_objs += 1
            if num_objs >= num_sources_in_query:
                break

    ulens_con.close()
    return objs


def calculate_delta_m(u0, b_sff):
    # delta_f = (a * f_S + f_LN) / (f_T)
    # delta_f = (a * b_sff * f_T + (1 - b_sff) * f_T) / (f_T)
    # delta_f = a * b_sff + (1 - b_sff) = 1 + (a - 1) * b_sff
    # delta_m = | -2.5 * np.log10(delta_f) |

    amp = np.abs((u0 ** 2 + 2) / (u0 * np.sqrt(u0 ** 2 + 4)))
    delta_f = 1 + (amp - 1) * b_sff
    delta_m = 2.5 * np.log10(delta_f)
    return delta_m


def generate_random_lightcurves_lb(l, b, objs,
                                   tE_min=20, delta_m_min=0.5,
                                   num_3sigma_cut=5):
    popsycle_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
    popsycle_catalog = Table.read(popsycle_fname, format='fits')
    N_samples = len(objs)

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

    lightcurves_regular = []
    lightcurves_ulens = []

    for i, obj in enumerate(objs):
        if i % 100 == 0:
            print('Constructing lightcurve %i / %i' % (i, N_samples))
        tE = tE_arr[i]
        piE_E = piE_E_arr[i]
        piE_N = piE_N_arr[i]
        u0 = u0_arr[i]
        b_sff = b_sff_arr[i]
        mag_src = np.median(obj.lightcurve.mag) - 2.5 * np.log10(b_sff)

        obj_t = obj.lightcurve.hmjd
        obj_mag = obj.lightcurve.mag
        obj_magerr = obj.lightcurve.magerr
        obj_flux = obj.lightcurve.flux
        obj_fluxerr = obj.lightcurve.fluxerr
        t0_min, t0_max = np.min(obj_t), np.max(obj_t)

        attempts = 0
        while True:
            attempts += 1
            if attempts >= 100:
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
            delta_m_snr = delta_m / obj_magerr
            if np.sum(delta_m_snr >= 3) >= num_3sigma_cut:
                lightcurves_ulens.append((obj_t, obj_mag_micro, obj_magerr))
                lightcurves_regular.append((obj_t, obj_mag, obj_magerr))
                break

    return lightcurves_regular, lightcurves_ulens