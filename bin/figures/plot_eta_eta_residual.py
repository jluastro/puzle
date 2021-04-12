#! /usr/bin/env python
"""
plot_eta_eta_residual.py
"""

import numpy as np
from puzle.utils import return_figures_dir
from puzle.eta import return_eta_arrs, return_eta_ulens_arrs

import matplotlib.pyplot as plt


def plot_eta_eta_residual(eta_arr=None, eta_residual_arr=None, eta_threshold_low_best=None,
                          eta_ulens_arr=None, eta_residual_ulens_arr=None, observable_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_eta_arrs()
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, observable_arr = return_eta_ulens_arrs()

    cond_obs = observable_arr == True

    # linear-linear
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(eta_arr, eta_residual_arr, mincnt=1, gridsize=25)
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].hist(eta_threshold_low_best, color='r',
               bins=50, histtype='step', density=True)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_title('ulens total')
    ax[1].hexbin(eta_ulens_arr, eta_residual_ulens_arr,
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('log(eta_residual)', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_eta_eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)

    # log-linear
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(np.log10(eta_arr), eta_residual_arr, mincnt=1, gridsize=25)
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].hist(np.log10(eta_threshold_low_best), color='r',
               bins=50, histtype='step', density=True)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_title('ulens total')
    ax[1].hexbin(np.log10(eta_ulens_arr), eta_residual_ulens_arr,
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(np.log10(eta_ulens_arr[cond_obs]), eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_log-eta_eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)

    # log-log
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(np.log10(eta_arr), np.log10(eta_residual_arr), mincnt=1, gridsize=25)
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].hist(np.log10(eta_threshold_low_best), color='r',
               bins=50, histtype='step', density=True)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_title('ulens total')
    ax[1].hexbin(np.log10(eta_ulens_arr), np.log10(eta_residual_ulens_arr),
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(np.log10(eta_ulens_arr[cond_obs]), np.log10(eta_residual_ulens_arr[cond_obs]),
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('log(eta_residual)', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_log-eta_log-eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_eta_eta_threshold(eta_arr=None, eta_threshold_low_best=None):
    if eta_arr is None:
        eta_arr, _, eta_threshold_low_best = return_eta_arrs()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(eta_arr, eta_threshold_low_best,
               s=1, alpha=.2)
    ax.plot(np.arange(3), color='k', alpha=.2)
    ax.set_xlabel('eta', fontsize=10)
    ax.set_ylabel('eta_threshold', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/cands_eta_eta_threshold.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_eta_residual_ulens_vs_actual(eta_residual_ulens_arr=None,
                                      eta_residual_actual_ulens_arr=None,
                                      observable_arr=None):
    if eta_residual_ulens_arr is None:
        _, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()

    x_min = np.min([np.min(eta_residual_ulens_arr), np.min(eta_residual_actual_ulens_arr)])
    x_max = np.max([np.min(eta_residual_ulens_arr), np.max(eta_residual_actual_ulens_arr)])
    x = np.linspace(x_min, x_max)

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for a in ax: a.clear()
    ax[0].set_title('ulens total : %i samples' % len(cond_obs))
    ax[0].hexbin(eta_residual_ulens_arr,
                 eta_residual_actual_ulens_arr,
                 gridsize=25, mincnt=1)
    ax[1].set_title('ulens observable : %i samples' % np.sum(cond_obs))
    ax[1].hexbin(eta_residual_ulens_arr[cond_obs],
                 eta_residual_actual_ulens_arr[cond_obs],
                 gridsize=25, mincnt=1)
    for a in ax:
        a.set_xlabel('measured eta_residual ulens', fontsize=10)
        a.set_ylabel('modeled eta_residual ulens', fontsize=10)
        a.plot(x, x, color='r', linewidth=1)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_residual_vs_actual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_plots():
    eta_arr, eta_residual_arr, eta_threshold_low_best = return_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()
    plot_eta_eta_residual(eta_arr=eta_arr,
                          eta_residual_arr=eta_residual_arr,
                          eta_threshold_low_best=eta_threshold_low_best,
                          eta_ulens_arr=eta_ulens_arr,
                          eta_residual_ulens_arr=eta_residual_ulens_arr,
                          observable_arr=observable_arr)
    plot_eta_eta_threshold(eta_arr=eta_arr,
                           eta_threshold_low_best=eta_threshold_low_best)
    plot_eta_residual_ulens_vs_actual(eta_residual_ulens_arr=eta_residual_ulens_arr,
                                      eta_residual_actual_ulens_arr=eta_residual_actual_ulens_arr,
                                      observable_arr=observable_arr)


if __name__ == '__main__':
    generate_all_plots()
