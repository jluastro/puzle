#! /usr/bin/env python
"""
generate_microlensing_templates.py
"""

import os
from astropy.io import fits
import numpy as np
from microlens.jlu.model import PSPL_Phot_Par_Param1


def return_refined_events(snr_cut=15, mag_lim=21.5):
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    filename = f'{dir_path_puzle}/data/refined_events_sample.fits'
    refined_events = fits.open(filename)[1].data
    num_events = len(refined_events)
    mag_src = np.array([np.max(refined_events[i]['ubv_R_app_S'])
                        for i in range(num_events)])
    snr = 5 * 10 ** ((mag_lim - mag_src) / 5)
    cond = snr >= snr_cut
    return refined_events[cond]


def return_microlensing_templates(N_samples=10, snr_cut=15, mag_lim=21.5):
    refined_events = return_refined_events(snr_cut=snr_cut,
                                           mag_lim=mag_lim)
    N_events = len(refined_events)
    N_samples = min(N_events, N_samples)

    idx_arr = np.random.choice(np.arange(N_events), N_samples, replace=False)

    t_obs = np.linspace(-500, 500, 1001)

    phot_templates = np.zeros((N_samples, len(t_obs)))
    for i, idx in enumerate(idx_arr):
        t0 = refined_events[idx]['t0']
        u0_amp = refined_events[idx]['u0']
        tE = refined_events[idx]['t_E']
        piE = refined_events[idx]['pi_E']
        theta = np.random.uniform(0, 2*np.pi)
        piE_E = piE * np.cos(theta)
        piE_N = piE * np.sin(theta)
        b_sff = refined_events[idx]['f_blend_r']
        mag_src = refined_events[idx]['ubv_R_app_S']
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
        phot_templates[i] = phot + phot_err

    return phot_templates
