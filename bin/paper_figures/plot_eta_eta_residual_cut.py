#! /usr/bin/env python
"""
plot_eta_eta_residual_cut.py
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from puzle.utils import return_figures_dir
from puzle.eta import return_cands_level2_eta_arrs, \
    is_observable_frac_slope_offset
from puzle.ulens import return_ulens_level2_eta_arrs, return_cond_BH
from puzle.models import CandidateLevel2
from puzle.cands import return_eta_residual_slope_offset


def return_kde(xdata, ydata, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack((xdata, ydata))
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f


def plot_eta_eta_residual_cut():
    eta_arr, eta_residual_arr, _ = return_cands_level2_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, _, \
    _, _, observable_arr = return_ulens_level2_eta_arrs()

    cond_obs = observable_arr == True
    xmin, xmax, ymin, ymax = 0, 1.6, 0, 2.75
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
    print('%s of %s CandidateLevel2s Remaining (slope = %.2f | offset = %.2f)' %
                 (format(num_candidates_cut, ','),
                  format(num_candidates, ','),
                  slope, offset))
    for a in ax: a.clear()
    print('uLens frac = %.0f%% | cands frac = %.1f%%' %
                    (100*ulens_is_observable_frac_obs,
                     100*cands_is_observable_frac))
    ax[0].contour(ulens_obs_xx, ulens_obs_yy, ulens_obs_f, cmap='viridis', levels=10)
    ax[0].scatter(eta_ulens_arr[cond_obs],
                  eta_residual_ulens_arr[cond_obs],
                  color='b', alpha=0.05, s=1,
                  label=r'Simulated $\mu$-lens')
    print('BH uLens frac = %.0f%% | cands frac = %.1f%%' %
                    (100*ulens_is_observable_BH_frac_obs,
                     100*cands_is_observable_frac))
    ax[1].contour(ulens_BH_xx, ulens_BH_yy, ulens_BH_f, cmap='plasma', levels=10)
    ax[1].scatter(eta_ulens_arr[cond_obs*cond_BH],
                  eta_residual_ulens_arr[cond_obs*cond_BH],
                  color='darkgreen', alpha=0.5, s=1,
                  label=r'Simulated $\mu$-lens BHs')
    print('tE >= 150, piE <= 0.08')
    # ax[1].legend(markerscale=10, loc=4, fontsize=12)
    x = np.linspace(0, xmax)
    y = x * slope + offset
    for i, a in enumerate(ax):
        a.plot(x, y, color='k')
        a.contour(cands_xx, cands_yy, cands_f, cmap='autumn', levels=10)
        a.scatter(eta_arr, eta_residual_arr,
                  color='gold', alpha=0.01, s=1,
                  label='ZTF Candidates')
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel(r'$\eta$')
        a.set_ylabel(r'$\eta_{\rm residual}$')
        leg = a.legend(loc=4, markerscale=10)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/eta_eta_residual_cut.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_eta_eta_residual_cut()
