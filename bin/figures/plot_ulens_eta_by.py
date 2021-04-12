#! /usr/bin/env python
"""
plot_ulens_eta_by.py
"""

import numpy as np
from puzle.stats import calculate_eta_on_daily_avg, average_xy_on_round_x
from puzle.utils import load_stacked_array, return_data_dir, return_figures_dir
from puzle.eta import return_eta_ulens_arrs

import matplotlib.pyplot as plt


def plot_ulens_tE_piE(observable_arr=None):
    if observable_arr is None:
        _, _, _, observable_arr = return_eta_ulens_arrs()

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)

    cond = observable_arr == True

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    tE_bins = np.logspace(np.log10(2e1), 3, 15)
    ax[0].hist(tE, bins=tE_bins, histtype='step', label='ulens total')
    ax[0].hist(tE[cond], bins=tE_bins, histtype='step', label='ulens obs')
    ax[0].set_xlabel('t_E', fontsize=12)
    ax[0].set_yscale('log')
    ax[0].legend()

    piE_bins = np.logspace(-2, np.log10(3), 15)
    ax[1].hist(piE, bins=piE_bins, histtype='step', label='ulens total')
    ax[1].hist(piE[cond], bins=piE_bins, histtype='step', label='ulens obs')
    ax[1].set_xlabel('pi_E', fontsize=12)
    ax[1].set_yscale('log')
    ax[1].legend()

    tE_counts, _ = np.histogram(tE, bins=tE_bins)
    tE_obs_counts, _ = np.histogram(tE[cond], bins=tE_bins)
    tE_obs_frac = tE_obs_counts / tE_counts
    ax[2].plot(tE_bins[:-1], tE_obs_frac, marker='.')
    ax[2].set_xlabel('t_E', fontsize=12)
    ax[2].set_ylabel('N_obs / N_total')

    piE_counts, _ = np.histogram(piE, bins=piE_bins)
    piE_obs_counts, _ = np.histogram(piE[cond], bins=piE_bins)
    piE_obs_frac = piE_obs_counts / piE_counts
    ax[3].plot(piE_bins[:-1], piE_obs_frac, marker='.')
    ax[3].set_xlabel('pi_E', fontsize=12)
    ax[3].set_ylabel('N_obs / N_total')

    for a in ax:
        a.set_xscale('log')
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_tE_piE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_tE_piE_vs_eta(eta_ulens_arr=None, eta_residual_ulens_arr=None, observable_arr=None):
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, observable_arr = return_eta_ulens_arrs()

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)

    cond = observable_arr == True

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    def return_fit(x, y):
        idx_arr = np.argsort(x)
        x_sort, y_sort = x[idx_arr], y[idx_arr]
        x_log, y_log = np.log10(x_sort), np.log10(y_sort)
        m, b = np.polyfit(x_log, y_log, deg=1)
        return 10 ** (x_log), 10 ** (x_log * m + b)

    ax[0].scatter(tE, eta_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[0].scatter(tE[cond], eta_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[0].plot(*return_fit(tE[cond], eta_ulens_arr[cond]), color='r', alpha=.4)
    ax[0].set_xlabel('tE')
    ax[0].set_ylabel('eta')

    ax[1].scatter(tE, eta_residual_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[1].scatter(tE[cond], eta_residual_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[1].plot(*return_fit(tE[cond], eta_residual_ulens_arr[cond]), color='r', alpha=.4)
    ax[1].set_xlabel('tE')
    ax[1].set_ylabel('eta_residual')

    ax[2].scatter(piE, eta_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[2].scatter(piE[cond], eta_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[2].plot(*return_fit(piE[cond], eta_ulens_arr[cond]), color='r', alpha=.4)
    ax[2].set_xlabel('piE')
    ax[2].set_ylabel('eta')

    ax[3].scatter(piE, eta_residual_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[3].scatter(piE[cond], eta_residual_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[3].plot(*return_fit(piE[cond], eta_residual_ulens_arr[cond]), color='r', alpha=.4)
    ax[3].set_xlabel('piE')
    ax[3].set_ylabel('eta_residual')

    for a in ax:
        a.legend(markerscale=3)
        a.set_xscale('log')
    fig.tight_layout()

    fname = '%s/ulens_tE_piE_vs_eta.png' % return_figures_dir()
    fig.savefig(fname)
    plt.close(fig)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_eta_by_mag(eta_ulens_arr=None,
                          eta_residual_ulens_arr=None, observable_arr=None):
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, observable_arr = return_eta_ulens_arrs()

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    mag_src = metadata['mag_src']

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable mag_src')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    mag_src_bins = [0, 17.5, 18.5, 19.5, 20.5, 21.5]
    for i in range(5):
        mag_src_low = mag_src_bins[i]
        mag_src_high = mag_src_bins[i+1]
        cond_mag_src = (mag_src > mag_src_low) * (mag_src <= mag_src_high)
        ax[i+1].set_title('%.1f < mag_src <= %.1f' % (mag_src_low, mag_src_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_mag_src],
                       eta_residual_ulens_arr[cond_obs*cond_mag_src],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_mag.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_eta_by_tE(eta_ulens_arr=None, eta_residual_ulens_arr=None, observable_arr=None):
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, observable_arr = return_eta_ulens_arrs()

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    tE = metadata['tE']

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable tE')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    tE_bins = [20, 50, 80, 120, 150, 1000]
    for i in range(5):
        tE_low = tE_bins[i]
        tE_high = tE_bins[i+1]
        cond_tE = (tE > tE_low) * (tE <= tE_high)
        ax[i+1].set_title('%.1f < tE <= %.1f' % (tE_low, tE_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_tE],
                       eta_residual_ulens_arr[cond_obs*cond_tE],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_tE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_eta_by_piE(eta_ulens_arr=None, eta_residual_ulens_arr=None, observable_arr=None):
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, observable_arr = return_eta_ulens_arrs()

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable piE')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    piE_bins = np.logspace(-2, np.log10(3), 6)
    for i in range(5):
        piE_low = piE_bins[i]
        piE_high = piE_bins[i+1]
        cond_piE = (piE > piE_low) * (piE <= piE_high)
        ax[i+1].set_title('%.2f < piE <= %.2f' % (piE_low, piE_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_piE],
                       eta_residual_ulens_arr[cond_obs*cond_piE],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_piE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_lowest_ulens_eta(eta_ulens_arr=None, observable_arr=None):
    if eta_ulens_arr is None:
        eta_ulens_arr, _, _, observable_arr = return_eta_ulens_arrs()

    cond = observable_arr == True
    cond_idx = np.where(cond == True)[0]
    eta_ulens_obs_arr = eta_ulens_arr[cond]

    idx_arr = np.argsort(eta_ulens_obs_arr)[:8]

    fname = '%s/ulens_sample.total.npz' % return_data_dir()
    data = load_stacked_array(fname)

    fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, idx in enumerate(idx_arr):
        hmjd = data[cond_idx[idx]][:, 0]
        mag = data[cond_idx[idx]][:, 1]
        eta_new = calculate_eta_on_daily_avg(hmjd, mag)
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].set_title('eta = %.3f | %.3f' % (eta_new, eta_ulens_obs_arr[idx]))
        ax[i].scatter(hmjd, mag, color='b', s=5)
        ax[i].scatter(hmjd_round, mag_round, color='g', s=5)
    for a in ax:
        a.set_xlabel('hmjd', fontsize=10)
        a.set_ylabel('mag', fontsize=10)
        a.invert_yaxis()
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_lowest_eta.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_plots():
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()
    plot_ulens_tE_piE(observable_arr=observable_arr)
    plot_ulens_tE_piE_vs_eta(eta_ulens_arr=eta_ulens_arr,
                             eta_residual_ulens_arr=eta_residual_ulens_arr,
                             observable_arr=observable_arr)
    plot_ulens_eta_by_mag(eta_ulens_arr=eta_ulens_arr,
                          eta_residual_ulens_arr=eta_residual_ulens_arr,
                          observable_arr=observable_arr)
    plot_ulens_eta_by_tE(eta_ulens_arr=eta_ulens_arr,
                         eta_residual_ulens_arr=eta_residual_ulens_arr,
                         observable_arr=observable_arr)
    plot_ulens_eta_by_piE(eta_ulens_arr=eta_ulens_arr,
                          eta_residual_ulens_arr=eta_residual_ulens_arr,
                          observable_arr=observable_arr)
    plot_lowest_ulens_eta(eta_ulens_arr=eta_ulens_arr,
                          observable_arr=observable_arr)


if __name__ == '__main__':
    generate_all_plots()
