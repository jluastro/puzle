#! /usr/bin/env python
"""
calculate_thresholds.py
"""

"""
For each field in the galactic plane:

- Select 150 lightcurves from each of 64 RCID for each filter (9,600 lightcurves for each filter)
    - Load RCID corners
    - Query "sources" database for sources
    - Grab first 150 lightcurves with:
        - At least 20 unique days observed in the filter
- Inject microlensing signals into a copy of all of the lightcurves
    - Find the nearest PopSyCLE simulation for population sourcing
    - Add microlensing from models.py with observable conditions
- Calculate thresholds
    - Calculate eta, chi-squared reduced and J on all lightcurves
    - For a FPR for each statistics ranging from 10% to 0.01%:
        - Determine the resulting False Negative rate for microlensing events
    - Save data to disk
"""

import matplotlib.pyplot as plt
import numpy as np

from puzle.utils import gather_PopSyCLE_lb, find_nearest_lightcurve_file
from puzle.sample import generate_random_lightcurves_lb, fetch_sample_objects, \
    calculate_lightcurve_stats


def generate_example_lightcurves(lightcurves_norm, lightcurves_ulens):
    fig, ax = plt.subplots(6, 3, figsize=(18, 9))
    axes = ax.flatten()
    idx_arr = np.random.choice(np.arange(len(lightcurves_norm)),
                               len(axes), replace=False)
    for i, ax in enumerate(axes):
        ax.clear()
        ax.errorbar(lightcurves_norm[idx_arr[i]][0],
                    lightcurves_norm[idx_arr[i]][1],
                    yerr=lightcurves_norm[idx_arr[i]][2],
                    color='r', linestyle='')
        ax.errorbar(lightcurves_ulens[idx_arr[i]][0],
                    lightcurves_ulens[idx_arr[i]][1],
                    yerr=lightcurves_ulens[idx_arr[i]][2],
                    color='b', linestyle='')
        ax.invert_yaxis()
    fig.tight_layout()


def generate_whelen_stats_figure(stats_norm, stats_ulens):
    eta_norm, J_norm, chi_squared_delta_norm = stats_norm
    eta_ulens, J_ulens, chi_squared_delta_ulens = stats_ulens

    eta_thresh = np.percentile(eta_norm, 1)
    J_thresh = np.percentile(J_norm, 99)
    chi_thresh = np.percentile(chi_squared_delta_norm, 99)

    fig, ax = plt.subplots(3, 2, figsize=(9, 18))
    size = 5
    alpha = 1
    for a in ax.flatten():
        a.clear()
        a.grid(True)
    ax[0, 0].set_xlabel('Eta', fontsize=12)
    ax[0, 0].set_ylabel('Chi-Squared', fontsize=12)
    ax[0, 0].scatter(eta_norm, chi_squared_delta_norm,
                     s=size, alpha=alpha)
    ax[0, 1].set_xlabel('Eta', fontsize=12)
    ax[0, 1].scatter(eta_ulens, chi_squared_delta_ulens,
                     s=size, alpha=alpha)
    for a in ax[0, :]:
        a.set_xlim(1e-2, 1e1)
        a.set_ylim(1e-4, 1e7)
        a.axvline(eta_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.axhline(chi_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.fill_between([1e-2, eta_thresh],
                       [chi_thresh, chi_thresh],
                       [1e7, 1e7],
                       color='r', alpha=.1)


    ax[1, 0].set_xlabel('Eta', fontsize=12)
    ax[1, 0].set_ylabel('J', fontsize=12)
    ax[1, 0].scatter(eta_norm, J_norm,
                     s=size, alpha=alpha)
    ax[1, 1].set_xlabel('Eta', fontsize=12)
    ax[1, 1].scatter(eta_ulens, J_ulens,
                     s=size, alpha=alpha)
    for a in ax[1, :]:
        a.set_xlim(1e-2, 1e1)
        a.set_ylim(1e-3, 1e5)
        a.axvline(eta_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.axhline(J_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.fill_between([1e-2, eta_thresh],
                       [J_thresh, J_thresh],
                       [1e5, 1e5],
                       color='r', alpha=.1)

    ax[2, 0].set_xlabel('Chi-Squared', fontsize=12)
    ax[2, 0].set_ylabel('J', fontsize=12)
    ax[2, 0].scatter(chi_squared_delta_norm, J_norm,
                     s=size, alpha=alpha)
    ax[2, 1].set_xlabel('Chi-Squared', fontsize=12)
    ax[2, 1].scatter(chi_squared_delta_ulens, J_ulens,
                     s=size, alpha=alpha)
    for a in ax[2, :]:
        a.set_xlim(1e-4, 1e7)
        a.set_ylim(1e-3, 1e5)
        a.axvline(chi_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.axhline(J_thresh, color='r',
                  alpha=.8, linestyle='--')
        a.fill_between([chi_thresh, 1e7],
                       [J_thresh, J_thresh],
                       [1e5, 1e5],
                       color='r', alpha=.1)

    for a in ax.flatten():
        a.set_xscale('log')
        a.set_yscale('log')
    fig.tight_layout()


lb_arr = gather_PopSyCLE_lb()
for i, (l, b) in enumerate(lb_arr):
    print('Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (l, b, i, len(lb_arr)))
    lightcurve_file = find_nearest_lightcurve_file(l, b)
    objs = fetch_sample_objects(lightcurve_file)
    lightcurves_norm, lightcurves_ulens = generate_random_lightcurves_lb(l, b, objs)

    stats_norm = calculate_lightcurve_stats(lightcurves_norm)
    eta_norm, J_norm, chi_squared_delta_norm = stats_norm
    stats_ulens = calculate_lightcurve_stats(lightcurves_ulens)
    eta_ulens, J_ulens, chi_squared_delta_ulens = stats_ulens
