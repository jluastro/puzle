#! /usr/bin/env python
"""
plot_eta_eta_residual.py
"""

import numpy as np
import scipy.stats as st
import copy
import glob

from puzle.utils import return_figures_dir, return_data_dir
from puzle.eta import return_level2_eta_arrs, return_eta_ulens_arrs, \
    is_observable_frac_slope_offset
from puzle.models import CandidateLevel2
from puzle.cands import return_eta_residual_slope_offset, \
    apply_eta_residual_slope_offset_to_query

import matplotlib
import matplotlib.pyplot as plt


def plot_eta_eta_residual(eta_arr=None, eta_residual_arr=None, eta_threshold_low_best=None,
                          eta_ulens_arr=None, eta_residual_ulens_arr=None, observable_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_level2_eta_arrs()
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, _, _, observable_arr = return_eta_ulens_arrs()

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
        eta_arr, _, eta_threshold_low_best = return_level2_eta_arrs()

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
        _, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, _,_, observable_arr = return_eta_ulens_arrs()

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


def return_kde(eta, eta_residual, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack((eta, eta_residual))
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f


def return_cond_BH(tE_min=150, piE_max=0.08):
    data_dir = return_data_dir()
    fname_total_arr = glob.glob(f'{data_dir}/ulens_sample_metadata.??.total.npz')
    fname_total_arr.sort()
    fname = fname_total_arr[-1]
    metadata = np.load(fname)
    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])
    cond_BH = tE >= tE_min
    cond_BH *= piE <= piE_max
    return cond_BH


def plot_eta_eta_residual_boundary_3obs(eta_arr=None, eta_residual_arr=None,
                                        eta_ulens_arr=None, eta_residual_ulens_arr=None,
                                        observable1_arr=None,
                                        observable2_arr=None,
                                        observable3_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_level2_eta_arrs()
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, \
        observable1_arr, observable2_arr, observable3_arr = return_eta_ulens_arrs()

    cond_obs1 = observable1_arr == True
    cond_obs2 = observable2_arr == True
    cond_obs3 = observable3_arr == True
    xmin, xmax, ymin, ymax = 0, 1.75, 0, 2.75
    bounds = (xmin, xmax, ymin, ymax)

    ulens1_xx, ulens1_yy, ulens1_f = return_kde(eta_ulens_arr[cond_obs1],
                                                eta_residual_ulens_arr[cond_obs1],
                                                *bounds)
    ulens2_xx, ulens2_yy, ulens2_f = return_kde(eta_ulens_arr[cond_obs2],
                                                eta_residual_ulens_arr[cond_obs2],
                                                *bounds)
    ulens3_xx, ulens3_yy, ulens3_f = return_kde(eta_ulens_arr[cond_obs3],
                                                eta_residual_ulens_arr[cond_obs3],
                                                *bounds)
    cands_xx, cands_yy, cands_f = return_kde(eta_arr,
                                             eta_residual_arr,
                                             *bounds)

    cond_BH = return_cond_BH()

    slope, offset = return_eta_residual_slope_offset()
    ulens_is_observable_frac_obs1 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs1],
                                                       eta_residual_ulens_arr[cond_obs1],
                                                       slope=slope, offset=offset)
    ulens_is_observable_frac_obs2 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs2],
                                                       eta_residual_ulens_arr[cond_obs2],
                                                       slope=slope, offset=offset)
    ulens_is_observable_frac_obs3 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs3],
                                                       eta_residual_ulens_arr[cond_obs3],
                                                       slope=slope, offset=offset)

    cands_is_observable_frac = is_observable_frac_slope_offset(eta_arr, eta_residual_arr,
                                                               slope=slope, offset=offset)

    ulens_is_observable_BH_frac_obs1 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs2*cond_BH],
                                                          eta_residual_ulens_arr[cond_obs2*cond_BH],
                                                          slope=slope, offset=offset)
    ulens_is_observable_BH_frac_obs2 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs2*cond_BH],
                                                          eta_residual_ulens_arr[cond_obs2*cond_BH],
                                                          slope=slope, offset=offset)
    ulens_is_observable_BH_frac_obs3 = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs3*cond_BH],
                                                          eta_residual_ulens_arr[cond_obs3*cond_BH],
                                                          slope=slope, offset=offset)

    num_candidates_cut = apply_eta_residual_slope_offset_to_query(CandidateLevel2.query).count()

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('ulens cond: 3-sigma / 3-increasing-brightness | slope: %.2f | offset: %.2f' %
                 (slope, offset), fontsize=10)
    for a in ax: a.clear()
    ax[0].set_title('1 point | uLens frac: %.1f%% | ulens BH frac: %.1f%% | cands frac: %.1f%%' %
                    (ulens_is_observable_frac_obs1,
                     ulens_is_observable_BH_frac_obs1,
                     cands_is_observable_frac),
                    fontsize=11)
    ax[0].contour(ulens1_xx, ulens1_yy, ulens1_f, cmap='viridis', levels=10)
    ax[0].scatter(eta_ulens_arr[cond_obs1],
                  eta_residual_ulens_arr[cond_obs1],
                  color='b', alpha=0.05, s=1)
    ax[1].set_title('2 points | uLens frac: %.1f%% | ulens BH frac: %.1f%% | cands frac: %.1f%%' %
                    (ulens_is_observable_frac_obs2,
                     ulens_is_observable_BH_frac_obs2,
                     cands_is_observable_frac),
                    fontsize=11)
    ax[1].contour(ulens2_xx, ulens2_yy, ulens2_f, cmap='viridis', levels=10)
    ax[1].scatter(eta_ulens_arr[cond_obs2],
                  eta_residual_ulens_arr[cond_obs2],
                  color='b', alpha=0.05, s=1)
    ax[2].set_title('3 points | uLens frac: %.1f%% | ulens BH frac: %.1f%% | cands frac: %.1f%%' %
                    (ulens_is_observable_frac_obs3,
                     ulens_is_observable_BH_frac_obs3,
                     cands_is_observable_frac),
                    fontsize=11)
    ax[2].contour(ulens3_xx, ulens3_yy, ulens3_f, cmap='viridis', levels=10)
    ax[2].scatter(eta_ulens_arr[cond_obs3],
                  eta_residual_ulens_arr[cond_obs3],
                  color='b', alpha=0.05, s=1)
    ax[2].scatter(eta_ulens_arr[cond_obs3*cond_BH],
                  eta_residual_ulens_arr[cond_obs3*cond_BH],
                  color='darkgreen', alpha=0.3, s=2)
    x = np.linspace(0, xmax)
    y = x * slope + offset
    for i, a in enumerate(ax):
        a.plot(x, y, color='k')
        a.contour(cands_xx, cands_yy, cands_f, cmap='autumn', levels=10)
        if i != 2:
            a.scatter(eta_arr, eta_residual_arr,
                      color='gold', alpha=0.01, s=1)
        else:
            a.scatter(eta_arr, eta_residual_arr,
                      color='gold', alpha=0.01, s=1,
                      label='%i candidates' % num_candidates_cut)
            a.legend()
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('eta', fontsize=10)
        a.set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_eta_eta_residual_boundary.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_eta_eta_residual_boundary(eta_arr=None, eta_residual_arr=None,
                                   eta_ulens_arr=None, eta_residual_ulens_arr=None,
                                   observable_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_level2_eta_arrs()
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, \
        _, _, observable_arr = return_eta_ulens_arrs()

    cond_obs = observable_arr == True
    xmin, xmax, ymin, ymax = 0, 1.75, 0, 2.75
    bounds = (xmin, xmax, ymin, ymax)

    ulens_obs_xx, ulens_obs_yy, ulens_obs_f = return_kde(eta_ulens_arr[cond_obs],
                                                         eta_residual_ulens_arr[cond_obs],
                                                         *bounds)
    cands_xx, cands_yy, cands_f = return_kde(eta_arr,
                                             eta_residual_arr,
                                             *bounds)

    cond_BH = return_cond_BH()
    ulens_BH_xx, ulens_BH_yy, ulens_BH_f = return_kde(eta_ulens_arr[cond_obs*cond_BH],
                                                      eta_residual_ulens_arr[cond_obs*cond_BH],
                                                      *bounds)

    slope, offset = return_eta_residual_slope_offset()
    ulens_is_observable_frac_obs = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs],
                                                      eta_residual_ulens_arr[cond_obs],
                                                      slope=slope, offset=offset)
    cands_is_observable_frac = is_observable_frac_slope_offset(eta_arr,eta_residual_arr,
                                                               slope=slope, offset=offset)

    ulens_is_observable_BH_frac_obs = is_observable_frac_slope_offset(eta_ulens_arr[cond_obs*cond_BH],
                                                         eta_residual_ulens_arr[cond_obs*cond_BH],
                                                         slope=slope, offset=offset)

    num_candidates = CandidateLevel2.query.count()
    num_candidates_cut = CandidateLevel2.query.filter(CandidateLevel2.eta_residual_best >=
                                                      CandidateLevel2.eta_best * slope + offset).\
                                            count()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('%s of %s CandidateLevel2s Remaining (slope = %.2f | offset = %.2f)' %
                 (format(num_candidates_cut, ','),
                  format(num_candidates, ','),
                  slope, offset), fontsize=10)
    for a in ax: a.clear()
    ax[0].set_title('uLens frac = %.1f%% | cands frac = %.2f%%' %
                    (100*ulens_is_observable_frac_obs,
                     100*cands_is_observable_frac),
                    fontsize=11)
    ax[0].contour(ulens_obs_xx, ulens_obs_yy, ulens_obs_f, cmap='viridis', levels=10)
    ax[0].scatter(eta_ulens_arr[cond_obs],
                  eta_residual_ulens_arr[cond_obs],
                  color='b', alpha=0.05, s=1)
    ax[1].set_title('BH uLens frac = %.1f%% | cands frac = %.2f%%' %
                    (100*ulens_is_observable_BH_frac_obs,
                     100*cands_is_observable_frac),
                    fontsize=11)
    ax[1].contour(ulens_BH_xx, ulens_BH_yy, ulens_BH_f, cmap='plasma', levels=10)
    ax[1].scatter(eta_ulens_arr[cond_obs*cond_BH],
                  eta_residual_ulens_arr[cond_obs*cond_BH],
                  color='darkgreen', alpha=0.5, s=1,
                  label='tE >= 150, piE <= 0.08')
    ax[1].legend(markerscale=10, loc=4, fontsize=12)
    x = np.linspace(0, xmax)
    y = x * slope + offset
    for i, a in enumerate(ax):
        a.plot(x, y, color='k')
        a.contour(cands_xx, cands_yy, cands_f, cmap='autumn', levels=10)
        a.scatter(eta_arr, eta_residual_arr,
                  color='gold', alpha=0.01, s=1)
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('eta', fontsize=10)
        a.set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_eta_eta_residual_boundary.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_eta_boundary_fracs(eta_arr=None,
                            eta_residual_arr=None,
                            eta_ulens_arr=None,
                            eta_residual_ulens_arr=None,
                            observable_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_level2_eta_arrs()
    if eta_ulens_arr is None:
        eta_ulens_arr, eta_residual_ulens_arr, _, _, _, observable_arr = return_eta_ulens_arrs()

    cond = observable_arr == True
    cond_BH = return_cond_BH()

    slope_arr = np.linspace(1, 7, 100)
    offset_arr = np.linspace(-0.5, 0.25, 100)
    slope_mesh, offset_mesh = np.meshgrid(slope_arr, offset_arr)
    frac_ulens_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    frac_ulens_BH_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    frac_cands_mesh = np.zeros((len(offset_mesh), len(slope_arr)))
    for i, offset in enumerate(offset_arr):
        for j, slope in enumerate(slope_arr):
            frac_ulens_mesh[i, j] = is_observable_frac_slope_offset(eta_ulens_arr[cond],
                                                                    eta_residual_ulens_arr[cond],
                                                                    offset=offset,
                                                                    slope=slope)
            frac_ulens_BH_mesh[i, j] = is_observable_frac_slope_offset(eta_ulens_arr[cond*cond_BH],
                                                                       eta_residual_ulens_arr[cond*cond_BH],
                                                                       offset=offset,
                                                                       slope=slope)
            frac_cands_mesh[i, j] = is_observable_frac_slope_offset(eta_arr,
                                                                    eta_residual_arr,
                                                                    offset=offset,
                                                                    slope=slope)
    logfrac_cands_mesh = np.log10(frac_cands_mesh)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Fraction Passing Cut')
    cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
    cmap.set_bad(color='red')
    for a in ax: a.clear()
    extent = (np.min(slope_arr), np.max(slope_arr),
              np.min(offset_arr), np.max(offset_arr))
    ax[0].set_title('All ulens Events', fontsize=12)
    im0 = ax[0].imshow(frac_ulens_mesh, origin='lower',
                       extent=extent, cmap=cmap, aspect='auto')
    cont0 = ax[0].contour(frac_ulens_mesh, levels=[0.75, 0.8, 0.9, 0.95, 0.99],
                          colors=['r', 'g', 'b', 'blueviolet', 'k'], origin='lower', extent=extent)
    cbar0 = fig.colorbar(im0, ax=ax[0], label='fraction passed')
    cbar0.add_lines(cont0)
    ax[1].set_title('BH ulens Events (tE >= 150 | piE <= 0.08)', fontsize=12)
    im1 = ax[1].imshow(frac_ulens_BH_mesh, origin='lower',
                       extent=extent, cmap=cmap, aspect='auto')
    cont1 = ax[1].contour(frac_ulens_BH_mesh, levels=[0.75, 0.8, 0.9, 0.95, 0.99],
                          colors=['r', 'g', 'b', 'blueviolet', 'k'], origin='lower', extent=extent)
    cbar1 = fig.colorbar(im1, ax=ax[1], label='fraction passed')
    cbar1.add_lines(cont1)
    ax[2].set_title('All CandidateLevel2s', fontsize=12)
    im2 = ax[2].imshow(logfrac_cands_mesh, origin='lower',
                       extent=extent, cmap=cmap, aspect='auto')
    cont2 = ax[2].contour(logfrac_cands_mesh, levels=[-2, -1.5, -1],
                          colors=['orange', 'tomato', 'orchid'], origin='lower', extent=extent)
    ax[1].contour(logfrac_cands_mesh, levels=[-2, -1.5, -1],
                  colors=['orange', 'tomato', 'orchid'], origin='lower', extent=extent)
    cbar2 = fig.colorbar(im2, ax=ax[2], label='LOG[fraction passed]')
    cbar2.add_lines(cont2)
    for a in ax:
        a.set_xlabel('slope', fontsize=10)
        a.set_ylabel('offset', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_eta_boundary_frac_v2.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_figures():
    eta_arr, eta_residual_arr, eta_threshold_low_best = return_level2_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, \
    observable1_arr, observable2_arr, observable3_arr = return_eta_ulens_arrs()
    plot_eta_eta_residual(eta_arr=eta_arr,
                          eta_residual_arr=eta_residual_arr,
                          eta_threshold_low_best=eta_threshold_low_best,
                          eta_ulens_arr=eta_ulens_arr,
                          eta_residual_ulens_arr=eta_residual_ulens_arr,
                          observable_arr=observable3_arr)
    plot_eta_eta_threshold(eta_arr=eta_arr,
                           eta_threshold_low_best=eta_threshold_low_best)
    plot_eta_residual_ulens_vs_actual(eta_residual_ulens_arr=eta_residual_ulens_arr,
                                      eta_residual_actual_ulens_arr=eta_residual_actual_ulens_arr,
                                      observable_arr=observable3_arr)
    plot_eta_eta_residual_boundary(eta_arr=eta_arr,
                                   eta_residual_arr=eta_residual_arr,
                                   eta_ulens_arr=eta_ulens_arr,
                                   eta_residual_ulens_arr=eta_residual_ulens_arr,
                                   observable_arr=observable3_arr)
    plot_eta_boundary_fracs(eta_arr=eta_arr,
                            eta_residual_arr=eta_residual_arr,
                            eta_ulens_arr=eta_ulens_arr,
                            eta_residual_ulens_arr=eta_residual_ulens_arr,
                            observable_arr=observable3_arr)


if __name__ == '__main__':
    generate_all_figures()
