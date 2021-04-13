#! /usr/bin/env python
"""
plot_cands_on_sky.py
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from puzle.models import Candidate, StarProcessJob, SourceIngestJob
from puzle.utils import return_figures_dir
from puzle import db


def plot_cands_on_sky():
    cands = Candidate.query.with_entities(Candidate.ra, Candidate.dec,
                                          Candidate.eta_best, Candidate.eta_residual_best).\
                      all()
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])

    slope = 3.57
    eta_thresh = 0.6
    cands_cut = [c for c in cands if c.eta_best <= eta_thresh and c.eta_residual_best > c.eta_best * slope]
    ra_arr_cut = np.array([c.ra for c in cands_cut])
    dec_arr_cut = np.array([c.dec for c in cands_cut])

    glon = np.linspace(0, 360, 10000)

    glat_low = np.zeros(len(glon)) - 20
    coords_low = SkyCoord(glon, glat_low, frame='galactic', unit='degree')
    ra_gal_low = coords_low.icrs.ra.value
    dec_gal_low = coords_low.icrs.dec.value

    glat_high = np.zeros(len(glon)) + 20
    coords_high = SkyCoord(glon, glat_high, frame='galactic', unit='degree')
    ra_gal_high = coords_high.icrs.ra.value
    dec_gal_high = coords_high.icrs.dec.value

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('slope = %.2f | eta_thresh = %.2f' % (slope, eta_thresh),
                 fontsize=10)
    for a in ax: a.clear()
    ax[0].set_title('All candidates', fontsize=12)
    ax[0].hexbin(ra_arr, dec_arr, gridsize=50, mincnt=1)
    ax[1].set_title('Cut candidates', fontsize=12)
    ax[1].hexbin(ra_arr_cut, dec_arr_cut, gridsize=50, mincnt=1)
    for a in ax:
        a.scatter(ra_gal_low, dec_gal_low, c='k', s=.1, alpha=.2)
        a.scatter(ra_gal_high, dec_gal_high, c='k', s=.1, alpha=.2)
        a.set_xlabel('ra', fontsize=12)
        a.set_ylabel('dec', fontsize=12)
        a.set_xlim(0, 360)
        a.set_ylim(-30, 90)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    fname = '%s/cands_on_sky.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_rf_pass_on_sky():
    jobs = db.session.query(SourceIngestJob, StarProcessJob).\
        filter(SourceIngestJob.id == StarProcessJob.source_ingest_job_id).\
        with_entities(SourceIngestJob.ra_start, SourceIngestJob.ra_end,
                      SourceIngestJob.dec_start, SourceIngestJob.dec_end,
                      StarProcessJob.num_stars_pass_n_days,
                      StarProcessJob.num_stars_pass_eta,
                      StarProcessJob.num_stars_pass_rf).\
        all()

    ra_arr = [(d[0]+d[1])/2 for d in jobs]
    dec_arr = [(d[2] + d[3]) / 2 for d in jobs]
    num_stars_pass_n_days_arr = np.array([d[4] for d in jobs])
    num_stars_pass_eta_arr = np.array([d[5] for d in jobs])
    num_stars_pass_rf_arr = np.array([d[6] for d in jobs])

    log_eta_pass_frac = np.log10(num_stars_pass_eta_arr / num_stars_pass_n_days_arr)
    log_rf_pass_frac = np.log10(num_stars_pass_rf_arr / num_stars_pass_eta_arr)

    glon = np.linspace(0, 360, 10000)

    glat_low = np.zeros(len(glon)) - 20
    coords_low = SkyCoord(glon, glat_low, frame='galactic', unit='degree')
    ra_gal_low = coords_low.icrs.ra.value
    dec_gal_low = coords_low.icrs.dec.value

    glat_high = np.zeros(len(glon)) + 20
    coords_high = SkyCoord(glon, glat_high, frame='galactic', unit='degree')
    ra_gal_high = coords_high.icrs.ra.value
    dec_gal_high = coords_high.icrs.dec.value

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Survival Rates on Sky', fontsize=10)
    for a in ax: a.clear()
    ax[0].set_title('Eta Cut', fontsize=12)
    im0 = ax[0].scatter(ra_arr, dec_arr, c=log_eta_pass_frac, s=2)
    cbar0 = fig.colorbar(im0, ax=ax[0], label='log(eta pass frac)')
    ax[1].set_title('star/galaxy Cut', fontsize=12)
    im1 = ax[1].scatter(ra_arr, dec_arr, c=log_rf_pass_frac, s=2)
    cbar1 = fig.colorbar(im1, ax=ax[1], label='log(rf pass frac)')
    for a in ax:
        a.scatter(ra_gal_low, dec_gal_low, c='k', s=.1, alpha=.2)
        a.scatter(ra_gal_high, dec_gal_high, c='k', s=.1, alpha=.2)
        a.set_xlabel('ra', fontsize=12)
        a.set_ylabel('dec', fontsize=12)
        a.set_xlim(0, 360)
        a.set_ylim(-30, 90)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    fname = '%s/rf_pass_on_sky.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_figures():
    plot_cands_on_sky()


if __name__ == '__main__':
    generate_all_figures()
