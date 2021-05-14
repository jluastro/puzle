#! /usr/bin/env python
"""
plot_job_cells.py
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pickle
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.coordinates import SkyCoord

from puzle.models import SourceIngestJob, StarProcessJob
from puzle.utils import return_figures_dir, return_DR5_dir, return_data_dir
from puzle import db


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def _load_process_progress(type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_process_progress.dict'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return {}


def load_star_process_progress():
    return _load_process_progress('star')


def _save_process_progress(process_progress, type):
    data_dir = return_data_dir()
    fname = f'{data_dir}/{type}_process_progress.dict'
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pickle.dump(process_progress, f)


def save_star_process_progress(process_progress):
    _save_process_progress(process_progress, 'star')


def plot_job_cells():
    DR5_dir = return_DR5_dir()
    jobs = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id).\
        all()

    star_process_progress = load_star_process_progress()

    num_stars_arr = []
    ra_arr = []
    dec_arr = []
    num_objs_arr = []
    datetime_delta_arr = []
    for i, job in enumerate(jobs):
        if i % 1000 == 0:
            print(i, len(jobs))
        job_id = job[0].id
        if job[1].num_objs is None:
            num_objs = 0
        else:
            num_objs = job[1].num_objs

        if job_id in star_process_progress:
            ra = star_process_progress[job_id][0]
            dec = star_process_progress[job_id][1]
            num_stars = star_process_progress[job_id][2]
        else:
            ra = (job[0].ra_start + job[0].ra_end) / 2
            dec = (job[0].dec_start + job[0].dec_end) / 2
            dir = '%s/stars_%s' % (DR5_dir, str(job_id)[:3])
            fname = f'{dir}/stars.{job_id:06}.txt'
            num_stars = file_len(fname)
            star_process_progress[job_id] = (ra, dec, num_stars)

        datetime_delta = job[1].datetime_finished - job[1].datetime_started
        datetime_delta_arr.append(datetime_delta.seconds)

        num_stars_arr.append(num_stars)
        num_objs_arr.append(num_objs)
        ra_arr.append(ra)
        dec_arr.append(dec)

    save_star_process_progress(star_process_progress)

    ra_arr = np.array(ra_arr)
    dec_arr = np.array(dec_arr)
    num_stars_arr = np.array(num_stars_arr)
    num_objs_arr = np.array(num_objs_arr)
    datetime_delta_arr = np.array(datetime_delta_arr)

    ra_bins = np.arange(0, 361, 2)
    dec_bins = np.arange(-30, 91, 1)
    extent = (ra_bins.min(), ra_bins.max(), dec_bins.min(), dec_bins.max())
    job_hist = np.histogram2d(ra_arr, dec_arr, bins=(ra_bins, dec_bins))[0].T
    num_objs_hist = binned_statistic_2d(ra_arr, dec_arr, num_objs_arr,
                                        statistic='sum', bins=(ra_bins, dec_bins))[0].T

    glon = np.linspace(0, 360, 10000)
    
    glat_low = np.zeros(len(glon)) - 20
    coords_low = SkyCoord(glon, glat_low, frame='galactic', unit='degree')
    ra_gal_low = coords_low.icrs.ra.value
    dec_gal_low = coords_low.icrs.dec.value

    glat_high = np.zeros(len(glon)) + 20
    coords_high = SkyCoord(glon, glat_high, frame='galactic', unit='degree')
    ra_gal_high = coords_high.icrs.ra.value
    dec_gal_high = coords_high.icrs.dec.value

    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    for a in ax: a.clear()

    ax[0].set_title('Star Process Jobs')
    im0 = ax[0].imshow(job_hist / 2, extent=extent, origin='lower')
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.set_label(r'Number of Jobs / deg$^2$', fontsize=12)

    ax[1].set_title(r'Objects with $n_{\rm epochs} \geq 20$')
    norm = LogNorm(vmin=1e4, vmax=1e6)
    im0 = ax[1].imshow(num_objs_hist / 2, norm=norm, extent=extent, origin='lower')
    cbar0 = fig.colorbar(im0, ax=ax[1])
    cbar0.set_label(r'Number of Objects / deg$^2$', fontsize=12)

    for a in ax:
        a.scatter(ra_gal_low, dec_gal_low, c='k', s=.1, alpha=.05)
        a.scatter(ra_gal_high, dec_gal_high, c='k', s=.1, alpha=.05)
        a.set_xlabel('ra', fontsize=12)
        a.set_ylabel('dec', fontsize=12)
        a.set_xlim(0, 360)
        a.set_ylim(-28, 90)
    fig.tight_layout()

    fname = '%s/job_cells.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_job_cells()
