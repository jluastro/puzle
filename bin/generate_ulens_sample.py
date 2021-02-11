import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from zort.lightcurveFile import LightcurveFile
from zort.photometry import fluxes_to_magnitudes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.utils import fetch_lightcurve_rcids, return_figures_dir, lightcurve_file_to_field_id

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

        tables = []
        for fi in fis:
            t = Table.read(fi, format='fits')
            N_samples += len(t)
            tables.append(t)
        table_new = vstack(tables)

        fi_new = f'{popsycle_base_folder}/{lb}_refined_events_ztf_r_Damineli16.fits'
        table_new.write(fi_new, overwrite=True)

    print(f'{N_samples} Samples')


def generate_random_lightcurves_lb(l, b, N_samples=1000, N_t0_samples=10, nepochs_min=20):
    popsycle_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
    popsycle_catalog = Table.read(popsycle_fname, format='fits')

    tE_log_catalog = np.log10(popsycle_catalog['t_E'])
    tE_log_median = np.median(tE_log_catalog)
    tE_log_std = np.std(tE_log_catalog)
    tE_arr = 10 ** np.random.normal(tE_log_median, tE_log_std, size=N_samples)

    pi_E_catalog = popsycle_catalog['pi_E']
    loc, scale = expon.fit(pi_E_catalog)
    pi_E = expon.rvs(loc, scale, N_samples)
    theta = np.random.uniform(0, 2 * np.pi, N_samples)
    piE_E_arr = pi_E * np.cos(theta)
    piE_N_arr = pi_E * np.sin(theta)

    u0_arr = np.random.uniform(-2, 2, N_samples)
    b_sff_arr = np.random.uniform(0, 1, N_samples)

    coord = SkyCoord(l, b, unit=u.degree, frame='galactic')
    ra, dec = coord.icrs.ra.value, coord.icrs.dec.value
    radius = np.sqrt(47 / np.pi)
    delta_x = radius * np.sqrt(2) / 2
    ra_start, ra_end = ra - delta_x, ra + delta_x
    dec_start, dec_end = dec - delta_x, dec + delta_x
    lightcurve_rcids = fetch_lightcurve_rcids(ra_start, ra_end, dec_start, dec_end)

    lightcurveFile_arr = []
    for lightcurve_filename, rcids in lightcurve_rcids:
        field_id = lightcurve_file_to_field_id(lightcurve_filename)
        if field_id > 1000:
            continue
        lightcurveFile = LightcurveFile(lightcurve_filename, rcids_to_read=rcids)
        lightcurveFile_arr.append(lightcurveFile)

    for i in range(N_samples):
        print('Generating Sample %i/%i' % (i, N_samples))
        obj1 = None
        obj2 = None
        lightcurveFile_idx = np.random.choice(np.arange(len(lightcurveFile_arr)))
        lightcurveFile = lightcurveFile_arr[lightcurveFile_idx]
        for obj in lightcurveFile:
            n_days = len(np.unique(np.floor(obj.lightcurve.hmjd)))
            print('-- %i' % n_days)
            if n_days >= nepochs_min:
                if obj1 is None:
                    obj1 = obj
                else:
                    obj2 = obj
            if obj2 is not None:
                break

        tE = tE_arr[i]
        piE_E = piE_E_arr[i]
        piE_N = piE_N_arr[i]
        u0 = u0_arr[i]
        b_sff = b_sff_arr[i]
        mag_src = np.median(obj1.lightcurve.mag) * b_sff

        obj_t = obj1.lightcurve.hmjd
        obj_flux = obj1.lightcurve.flux
        obj_fluxerr = obj1.lightcurve.fluxerr
        t0_min, t0_max = np.min(obj_t), np.max(obj_t)
        t0_arr = np.random.uniform(t0_min, t0_max, size=N_t0_samples)

        fig, ax = plt.subplots(5, 2, figsize=(8, 8))
        ax = ax.flatten()
        for j, t0 in enumerate(t0_arr):
            model = PSPL_Phot_Par_Param1(t0=t0, u0_amp=u0, tE=tE, mag_src=mag_src,
                                         piE_E=piE_E, piE_N=piE_N, b_sff=b_sff,
                                         raL=obj1.ra, decL=obj1.dec)
            # f_micro = A * f_source + f_neighbor_lens
            # f_neighbor_lens = f_total - f_source = f_total - b_sff * f_total = (1 - b_sff) * f_total
            # f_micro = A * b_sff * f_total + (1 - b_sff) * f_total

            amp = model.get_amplification(obj_t)
            obj_flux_micro = amp * b_sff * obj_flux + (1 - b_sff) * obj_flux
            obj_mag_micro, _ = fluxes_to_magnitudes(obj_flux_micro, obj_fluxerr)

            ax[j].scatter(obj_t, obj1.lightcurve.mag, s=50, marker='+', color='b')
            ax[j].scatter(obj_t, obj_mag_micro, s=50, marker='.', color='r')
            ax[j].invert_yaxis()
        fig.tight_layout()

        fname = '%s/ulens_lightcurve_sample_%04d.png' % (return_figures_dir(), i)
        fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
        print(f'{fname} saved')


if __name__ == '__main__':
    l, b, N_samples, N_t0_samples, nepochs_min = 6, 3, 1000, 10, 10
    generate_random_lightcurves_lb(l, b,
                                   N_samples=N_samples,
                                   N_t0_samples=N_t0_samples,
                                   nepochs_min=nepochs_min)