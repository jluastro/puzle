#! /usr/bin/env python
"""
generate_ulens_templates.py
"""

import matplotlib.pyplot as plt
import os
from astropy.io import fits
import numpy as np
from microlens.jlu.model import PSPL_Phot_Par_Param1


def return_refined_events_filename():
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    filename = f'{dir_path_puzle}/data/combined_5yrs_refined_events_ztf_r_Damineli16.v3_largeAp.fits'
    return filename


def return_refined_events(snr_cut=15, mag_lim=21.5):
    filename = return_refined_events_filename()
    refined_events = fits.open(filename)[1].data

    # perform cut on minimum SNR
    mag_src = refined_events['ztf_r_app_S']
    snr = 5 * 10 ** ((mag_lim - mag_src) / 5)
    cond = snr >= snr_cut
    refined_events = refined_events[cond]

    return refined_events


def return_ulens_templates(N_samples=500, snr_cut=15, mag_lim=21.5):
    refined_events = return_refined_events(snr_cut=snr_cut,
                                           mag_lim=mag_lim)
    N_events = len(refined_events)
    N_samples = min(N_events, N_samples)

    idx_arr = np.random.choice(np.arange(N_events), N_samples, replace=False)

    t_obs = np.linspace(-500, 500, 1001)

    ulens_templates = np.zeros((N_samples, len(t_obs)))
    for i, idx in enumerate(idx_arr):
        t0 = refined_events[idx]['t0']
        u0_amp = refined_events[idx]['u0']
        tE = refined_events[idx]['t_E']
        piE = refined_events[idx]['pi_E']
        theta = np.random.uniform(0, 2*np.pi)
        piE_E = piE * np.cos(theta)
        piE_N = piE * np.sin(theta)
        b_sff = refined_events[idx]['f_blend_r']
        mag_src = refined_events[idx]['ztf_r_app_S']
        raL = np.random.uniform(0, 360)
        decL = np.random.uniform(-30, 90)

        model = PSPL_Phot_Par_Param1(t0=t0,
                                     u0_amp=u0_amp,
                                     tE=tE,
                                     piE_E=piE_E,
                                     piE_N=piE_N,
                                     b_sff=b_sff,
                                     mag_src=mag_src,
                                     raL=raL,
                                     decL=decL)

        phot = model.get_photometry(t_obs)
        snr = 5 * 10 ** ((mag_lim - phot) / 5)
        phot_err = np.array([np.random.normal(scale=1/s) for s in snr])
        # ulens_templates[i] = phot + phot_err
        ulens_templates[i] = phot

    return ulens_templates


def plot_ulens_templates():
    ulens_templates = return_ulens_templates()
    N_templates = len(ulens_templates)
    N_samples = min(N_templates, 24)
    idx_arr = np.random.choice(np.arange(N_templates), N_samples, replace=False)

    fig, ax = plt.subplots(8, 3, figsize=(8, 10))
    ax = ax.flatten()

    for i, idx in enumerate(idx_arr):
        ulens_template = ulens_templates[idx]
        x = np.arange(len(ulens_template))
        ax[i].scatter(x, ulens_template, s=1)
        ax[i].ticklabel_format(useOffset=False)
        ax[i].invert_yaxis()

    fig.tight_layout()
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    fname = f'{dir_path_puzle}/figures/ulens_templates.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight',
                pad_inches=0.01)