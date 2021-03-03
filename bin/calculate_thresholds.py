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
import os

from puzle.utils import gather_PopSyCLE_lb, find_nearest_lightcurve_file, \
    return_data_dir, return_figures_dir
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


def generate_whelen_stats_figure(l, b, eta_norm, J_norm, chi_norm,
                                 eta_ulens, J_ulens, chi_ulens):

    eta_thresh = np.percentile(eta_norm, 1)
    J_thresh = np.percentile(J_norm, 99)
    chi_thresh = np.percentile(chi_norm, 99)

    fig, ax = plt.subplots(3, 2, figsize=(9, 18))
    size = 2
    alpha = .2
    for a in ax.flatten():
        a.clear()
        a.grid(True)
    ax[0, 0].set_xlabel('Eta', fontsize=12)
    ax[0, 0].set_ylabel('Chi-Squared', fontsize=12)
    ax[0, 0].scatter(eta_norm, chi_norm,
                     s=size, alpha=alpha)
    ax[0, 1].set_xlabel('Eta', fontsize=12)
    ax[0, 1].scatter(eta_ulens, chi_ulens,
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
    ax[2, 0].scatter(chi_norm, J_norm,
                     s=size, alpha=alpha)
    ax[2, 1].set_xlabel('Chi-Squared', fontsize=12)
    ax[2, 1].scatter(chi_ulens, J_ulens,
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

    fname = '%s/l%.1f_b%.1f_thresholds.png' % (return_figures_dir(), l, b)
    fig.savefig(fname, dpi=100, bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)


def generate_whelen_stats_figures():
    lb_arr = gather_PopSyCLE_lb()
    for (l, b) in lb_arr:
        data = np.load(fname)
        eta_ulens = data['eta_ulens']
        J_ulens = data['J_ulens']
        chi_ulens = data['chi_ulens']
        eta_norm = data['eta_norm']
        J_norm = data['J_norm']
        chi_norm = data['chi_norm']
        generate_whelen_stats_figure(l, b, eta_norm, J_norm, chi_norm,
                                     eta_ulens, J_ulens, chi_ulens)


def save_threshold_stats(overwrite=False):
    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank, size = 0, 1

    lb_arr = gather_PopSyCLE_lb()
    my_lb_arr = np.array_split(lb_arr, size)[rank]

    for i, (l, b) in enumerate(my_lb_arr):
        fname = '%s/l%.1f_b%.1f_threshold_stats.npz' % (return_data_dir(), l, b)
        if not overwrite and os.path.exists(fname):
            continue
        print('%i) Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (rank, l, b, i, len(my_lb_arr)))
        lightcurve_file = find_nearest_lightcurve_file(l, b)
        objs = fetch_sample_objects(lightcurve_file)
        lightcurves_norm, lightcurves_ulens = generate_random_lightcurves_lb(l, b, objs)

        stats_norm = calculate_lightcurve_stats(lightcurves_norm)
        eta_norm, J_norm, chi_norm = stats_norm
        stats_ulens = calculate_lightcurve_stats(lightcurves_ulens)
        eta_ulens, J_ulens, chi_ulens = stats_ulens

        np.savez(fname,
                 eta_ulens=eta_ulens,
                 J_ulens=J_ulens,
                 chi_ulens=chi_ulens,
                 eta_norm=eta_norm,
                 J_norm=J_norm,
                 chi_norm=chi_norm)


def load_threshold_stats():
    threshold_stats = {}
    lb_arr = gather_PopSyCLE_lb()
    for i, (l, b) in enumerate(lb_arr):
        print('Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (l, b, i, len(lb_arr)))
        fname = '%s/l%.1f_b%.1f_threshold_stats.npz' % (return_data_dir(), l, b)
        data = np.load(fname)
        eta_ulens = data['eta_ulens']
        J_ulens = data['J_ulens']
        chi_ulens = data['chi_ulens']
        eta_norm = data['eta_norm']
        J_norm = data['J_norm']
        chi_norm = data['chi_norm']

        eta_thresh = np.percentile(eta_norm, 1)
        J_thresh = np.percentile(J_norm, 99)
        chi_thresh = np.percentile(chi_norm, 99)

        eta_tpr = np.sum(eta_ulens <= eta_thresh) / len(eta_ulens)
        J_tpr = np.sum(J_ulens >= J_thresh) / len(J_ulens)
        chi_tpr = np.sum(chi_ulens >= chi_thresh) / len(chi_ulens)

        threshold_stats[(l, b)] = {
            'eta_thresh': eta_thresh,
            'J_thresh': J_thresh,
            'chi_thresh': chi_thresh,
            'eta_tpr': eta_tpr,
            'J_tpr': J_tpr,
            'chi_tpr': chi_tpr}

    return threshold_stats


if __name__ == '__main__':
    save_threshold_stats()
