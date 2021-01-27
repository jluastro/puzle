#! /usr/bin/env python
"""
generate_ulens_templates.py
"""

import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from microlens.jlu.model import PSPL_Phot_Par_Param1, PSPL_Phot_Par_GP_Param1


def return_refined_events_filename():
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    # /global/cfs/cdirs/uLens/PopSyCLE_runs/PopSyCLE_runs_v3/l45.2_b4.9
    filename = f'{dir_path_puzle}/data/combined_5yrs_refined_events_ztf_r_Damineli16.fits'
    return filename


def return_refined_events(snr_cut=15, mag_lim=20.5):
    filename = return_refined_events_filename()
    refined_events = fits.open(filename)[1].data

    # perform cut on minimum delta_m
    delta_m = refined_events['delta_m_r']
    cond1 = delta_m >= .1

    # perform cut on minimum SNR
    mag_app = refined_events['ztf_r_app_LSN']
    snr = 5 * 10 ** ((mag_lim - mag_app) / 5)
    cond2 = snr >= snr_cut

    refined_events = refined_events[cond1 * cond2]

    return refined_events


def add_magnitudes(mag1, mag2):
    flux1 = 10 ** (mag1 / -2.5)
    flux2 = 10 ** (mag2 / -2.5)
    return -2.5 * np.log10(flux1 + flux2)


def return_ulens_templates(N_samples=500, snr_cut=15, mag_lim=20.5):
    refined_events = return_refined_events(snr_cut=snr_cut,
                                           mag_lim=mag_lim)
    N_events = len(refined_events)
    N_samples = min(N_events, N_samples)

    np.random.seed(42)
    idx_arr = np.random.choice(np.arange(N_events), N_samples, replace=False)

    t_obs = np.linspace(-1000, 1000, 2001)

    ulens_templates = np.zeros((N_samples, len(t_obs)))
    ulens_templates_with_err = np.zeros((N_samples, len(t_obs)))
    snr_arr = []
    delta_m_arr = []
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
        glon = refined_events[idx]['glon_L']
        glat = refined_events[idx]['glat_L']
        coord = SkyCoord(glon, glat, frame='galactic', unit=u.degree)
        raL = coord.icrs.ra.value
        decL = coord.icrs.dec.value

        # # priors from Nate's paper
        # gp_log_sigma = [np.random.normal(0, 5)]
        # gp_log_rho = [np.random.normal(0, 5)]
        # gp_log_So = [np.random.normal(0, 5)]
        # gp_log_omegao = [np.random.normal(0, 5)]
        # model = PSPL_Phot_Par_GP_Param1(
        #     t0=t0, u0_amp=u0_amp, tE=tE,
        #     piE_E=piE_E, piE_N=piE_N,
        #     b_sff=b_sff, mag_src=mag_src,
        #     raL=raL, decL=decL,
        #     gp_log_sigma=gp_log_sigma, gp_log_rho=gp_log_rho,
        #     gp_log_So=gp_log_So, gp_log_omegao=gp_log_omegao)
        # mag = model.get_photometry(t_obs)
        # snr = 5 * 10 ** ((mag_lim - mag) / 5)
        # mag_err = np.array([np.random.normal(scale=1/s) for s in snr])
        # mag, mag_err = model.get_photometry_with_gp(t_obs, mag, mag_err)

        model = PSPL_Phot_Par_Param1(
            t0=t0, u0_amp=u0_amp, tE=tE,
            piE_E=piE_E, piE_N=piE_N,
            b_sff=b_sff, mag_src=mag_src,
            raL=raL, decL=decL)
        mag = model.get_photometry(t_obs)
        snr = 5 * 10 ** ((mag_lim - mag) / 5)
        mag_err = np.array([np.random.normal(scale=1/s) for s in snr])

        ulens_templates_with_err[i] = mag + mag_err
        ulens_templates[i] = mag
        snr_arr.append(np.min(snr))
        delta_m_arr.append(refined_events[idx]['delta_m_r'])

    return ulens_templates, ulens_templates_with_err


def save_ulens_templates():
    _, ulens_templates_with_err = return_ulens_templates()
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    fname = f'{dir_path_puzle}/data/ulens_templates.npy'
    np.save(fname, ulens_templates_with_err)

    return ulens_templates_with_err


def plot_ulens_templates(ulens_templates_with_err=None):
    if ulens_templates_with_err is None:
        _, ulens_templates_with_err = return_ulens_templates()
    N_templates = len(ulens_templates_with_err)
    N_samples = min(N_templates, 24)
    np.random.seed(42)
    idx_arr = np.random.choice(np.arange(N_templates), N_samples, replace=False)

    fig, ax = plt.subplots(8, 3, figsize=(8, 10))
    ax = ax.flatten()

    for i, idx in enumerate(idx_arr):
        x = np.arange(len(ulens_templates_with_err[idx]))
        ax[i].scatter(x, ulens_templates_with_err[idx], s=1, alpha=.5)
        ax[i].ticklabel_format(useOffset=False)
        ax[i].invert_yaxis()

    fig.tight_layout()
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    fname = f'{dir_path_puzle}/figures/ulens_templates.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight',
                pad_inches=0.01)


if __name__ == '__main__':
    ulens_templates_with_err = save_ulens_templates()
    plot_ulens_templates(ulens_templates_with_err)
