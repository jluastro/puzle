#! /usr/bin/env python
"""
plot_pspl_gp_fit_cuts.py
"""

import matplotlib.pyplot as plt
import numpy as np

from puzle.models import CandidateLevel4


def plot_pspl_gp_fit_cuts():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        all()

    keys = ['t0',
            'u0_amp',
            'u0_amp_err',
            'tE',
            'tE_err',
            'b_sff',
            'b_sff_err',
            'piE_E',
            'piE_E_err',
            'piE_N',
            'piE_N_err',
            'piE',
            'piE_err',
            'rchi2']
    keys_err = [k for k in keys if f'{k}_err' in keys]

    data = {}
    for key in keys:
        data[key] = np.array([getattr(c, f'{key}_pspl_gp') for c in cands])
    data['cand_id'] = np.array([c.id for c in cands])

    error_frac = {}
    for key in keys_err:
        key_err = f'{key}_err'
        error_frac[key] = data[key_err] / data[key]

    error = {}
    for key in keys_err:
        key_err = f'{key}_err'
        error[key_err] = data[key_err]

    cond1 = error_frac['tE'] <= 0.2
    cond2 = data['u0_amp_err'] <= 0.1
    cond3 = data['piE_err'] <= 0.1
    cond4 = np.abs(data['u0_amp']) <= 1.0
    cond5 = data['b_sff'] <= 1.2
    cond6 = data['rchi2'] <= 3

    level5_cond = cond1 * cond2 * cond3 * cond4 * cond5 * cond6

    print('No filters', len(cond1), 'cands')
    print('Filters up to 1', np.sum(cond1), 'cands')
    print('Filters up to 2', np.sum(cond1 * cond2), 'cands')
    print('Filters up to 3', np.sum(cond1 * cond2 * cond3), 'cands')
    print('Filters up to 4', np.sum(cond1 * cond2 * cond3 * cond4), 'cands')
    print('Filters up to 5', np.sum(cond1 * cond2 * cond3 * cond4 * cond5), 'cands')
    print('Filters up to 6', np.sum(cond1 * cond2 * cond3 * cond4 * cond5 * cond6), 'cands')

    # fractional error hist
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, key in enumerate(keys_err):
        arr = error_frac[key]
        bins = np.linspace(-2, 2, 50)
        ax[i].hist(arr, bins=bins, histtype='step', density=True)
        ax[i].hist(arr[level5_cond], bins=bins, histtype='step', density=True)
        ax[i].set_xlabel(f'fractional error: {key}')
    fig.tight_layout()

    # error hist
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, key in enumerate(error.keys()):
        arr = error[key]
        bins = np.linspace(-2, 2, 50)
        # ax[i].hist(arr, bins=bins, histtype='step', density=True)
        ax[i].hist(arr[level5_cond], bins=bins, histtype='step', density=True, color='r')
        ax[i].set_xlabel(f'error: {key}')
    fig.tight_layout()

    # param hist
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    lims = [(-2, 2), (1, 3), (0, 2), (-2, 2), (-2, 2), (0, 2)]
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, key in enumerate(keys_err):
        arr = data[key]
        if key == 'tE':
            bins = np.logspace(lims[i][0], lims[i][1], 50)
            ax[i].set_xscale('log')
        else:
            bins = np.linspace(lims[i][0], lims[i][1], 50)
        ax[i].hist(arr, bins=bins, histtype='step', density=True)
        ax[i].hist(arr[level5_cond], bins=bins, histtype='step', density=True)
        ax[i].set_xlabel(key)
        if key == 'tE':
            ax[i].axvline(150, color='k', alpha=.5)
        elif key == 'piE':
            ax[i].axvline(0.08, color='k', alpha=.5)
    fig.tight_layout()

    # param vs fractional error
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    lims = [(-2, 2), (10, 500), (0, 2), (-2, 2), (-2, 2), (0, 2)]
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, key in enumerate(keys_err):
        arr_data = data[key]
        arr_error_frac = error_frac[key]
        ax[i].scatter(arr_data, arr_error_frac, s=1, alpha=.1)
        ax[i].scatter(arr_data[level5_cond], arr_error_frac[level5_cond], s=1, alpha=.1)
        ax[i].set_xlim(lims[i])
        ax[i].set_ylim(-2, 2)
        ax[i].grid(True)
        if key == 'tE':
            ax[i].set_xscale('log')
        ax[i].set_xlabel(key)
        if key == 'tE':
            ax[i].axvline(150, color='k', alpha=.5)
        elif key == 'piE':
            ax[i].axvline(0.08, color='k', alpha=.5)
    fig.tight_layout()


if __name__ == '__main__':
    plot_pspl_gp_fit_cuts()
